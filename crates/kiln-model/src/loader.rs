use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use safetensors::SafeTensors;

use kiln_core::config::ModelConfig;

use crate::quantized::{self, GptqConfig};
use crate::weights::*;

/// Weight name prefix for the language model within the VL checkpoint.
/// Qwen3.5-4B ships as a vision-language model; we only load the LM portion.
const LM_PREFIX: &str = "model.language_model.";

/// Safetensors index file for sharded checkpoints.
const INDEX_FILENAME: &str = "model.safetensors.index.json";
const GPTQ_CONFIG_FILENAME: &str = "quantize_config.json";
const MAX_SHARD_COUNT: usize = 4096;
const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SNAPSHOT_INPUT_BYTES: u64 = 16 * 1024 * 1024 * 1024 * 1024;
const SNAPSHOT_COPY_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const SNAPSHOT_FREE_SPACE_RESERVE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug)]
struct ImmutableModelSnapshot {
    lease: Arc<ModelSnapshotLease>,
}

impl ImmutableModelSnapshot {
    fn root(&self) -> &Path {
        self.lease.path()
    }
}

#[derive(Debug)]
struct ShardMetadata {
    path: std::path::PathBuf,
    size_bytes: u64,
    modified_ns: u128,
}

#[derive(Debug)]
struct LoadedShard {
    meta: Arc<ShardMetadata>,
    file: Arc<fs::File>,
    mmap: Arc<Mmap>,
}

#[derive(Debug)]
struct TensorMapEntry<'a> {
    shard: Arc<ShardMetadata>,
    mmap: Arc<Mmap>,
    view: safetensors::tensor::TensorView<'a>,
}

type TensorMap<'a> = HashMap<String, TensorMapEntry<'a>>;

/// Optional loader toggles for startup-sensitive callers.
#[derive(Debug, Clone, Copy)]
pub struct LoadModelOptions {
    /// Load the checkpoint's native MTP head when present.
    ///
    /// Startup-sensitive callers may set this to `false` and rely on the
    /// deferred MTP path instead, which keeps routing support visible while
    /// postponing the CPU load until the first actual native-MTP request.
    pub load_mtp: bool,
}

impl Default for LoadModelOptions {
    fn default() -> Self {
        Self { load_mtp: true }
    }
}

/// Build a progress bar for the per-layer load loop.
///
/// Returns `None` when stderr is not a TTY so log files and piped output stay
/// clean. The structured `tracing::info!` lines remain the source of truth for
/// non-interactive runs; the bar is purely additive UX for the first-run
/// `kiln serve` experience, where 32 layers stream off disk over 5–30s with no
/// other visual feedback.
fn make_layer_progress(num_layers: usize) -> Option<indicatif::ProgressBar> {
    if !console::Term::stderr().features().is_attended() {
        return None;
    }
    let pb = indicatif::ProgressBar::new(num_layers as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "  loading layers {bar:40.cyan/blue} {pos:>3}/{len:3} ({elapsed})",
        )
        .expect("static progress template is valid")
        .progress_chars("##-"),
    );
    Some(pb)
}

/// Load Qwen3.5-4B language model weights from a directory of safetensors files.
///
/// Handles both single-file (`model.safetensors`) and sharded checkpoints
/// (`model-00001-of-NNNNN.safetensors` with an index file).
///
/// The model directory should be a HuggingFace model download containing
/// the safetensors files. Only language model weights (prefixed with
/// `model.language_model.`) are loaded; vision encoder weights are skipped.
pub fn load_model(model_dir: &Path, config: &ModelConfig) -> Result<ModelWeights> {
    load_model_with_options(model_dir, config, LoadModelOptions::default())
}

/// Variant of [`load_model`] with explicit startup-path options.
pub fn load_model_with_options(
    model_dir: &Path,
    config: &ModelConfig,
    options: LoadModelOptions,
) -> Result<ModelWeights> {
    load_model_with_options_and_snapshot_dir(model_dir, config, options, None)
}

/// Load from a private immutable snapshot, optionally placing that snapshot
/// under an operator-selected directory. Server configuration resolves any
/// environment override before this boundary; the model crate does not read
/// ambient placement state.
pub fn load_model_with_options_and_snapshot_dir(
    model_dir: &Path,
    config: &ModelConfig,
    options: LoadModelOptions,
    snapshot_dir: Option<&Path>,
) -> Result<ModelWeights> {
    let snapshot = create_immutable_model_snapshot(model_dir, snapshot_dir).with_context(|| {
        format!(
            "failed to create immutable model snapshot from {}; configure a private snapshot directory on a filesystem with sufficient free space",
            model_dir.display()
        )
    })?;

    // Auto-detect GPTQ quantization.
    if let Some(gptq_config) = quantized::load_gptq_config(snapshot.root())? {
        tracing::info!(
            bits = gptq_config.bits,
            group_size = gptq_config.group_size,
            sym = gptq_config.sym,
            "Detected GPTQ quantization — loading quantized weights"
        );
        return load_model_gptq(
            snapshot.root(),
            config,
            &gptq_config,
            options,
            Arc::clone(&snapshot.lease),
        );
    }

    load_model_dense(
        snapshot.root(),
        config,
        options,
        Arc::clone(&snapshot.lease),
    )
}

/// Load dense (unquantized) BF16/FP16 model weights.
fn load_model_dense(
    model_dir: &Path,
    config: &ModelConfig,
    options: LoadModelOptions,
    snapshot_lease: Arc<ModelSnapshotLease>,
) -> Result<ModelWeights> {
    let shards = discover_shards(model_dir)?;
    tracing::info!(
        "Loading model from {} ({} shard{})",
        model_dir.display(),
        shards.len(),
        if shards.len() == 1 { "" } else { "s" }
    );

    // Memory-map all shards and parse safetensors headers.
    let loaded_shards = mmap_shards(&shards)?;
    let source_content_guard =
        loaded_shard_content_guard(&loaded_shards, Arc::clone(&snapshot_lease));
    let source_content_sha256 = source_content_guard.initial_sha256().to_string();
    let parsed: Vec<SafeTensors<'_>> = loaded_shards
        .iter()
        .map(|shard| {
            SafeTensors::deserialize(&shard.mmap[..]).context("Failed to parse safetensors file")
        })
        .collect::<Result<Vec<_>>>()?;

    let tensor_map = build_tensor_map(&parsed, &loaded_shards);

    // Auto-detect prefix: try "model.language_model." first, fall back to "model.".
    let prefix = detect_prefix(&tensor_map);
    tracing::info!("Using weight name prefix: \"{prefix}\"");

    // Load embedding.
    let embed_tokens = extract_tensor(&tensor_map, &format!("{prefix}embed_tokens.weight"))?;
    validate_shape(
        &embed_tokens,
        &[config.vocab_size, config.hidden_size],
        "embed_tokens",
    )?;

    // Load layers.
    let mut layers = Vec::with_capacity(config.num_layers);
    let pb = make_layer_progress(config.num_layers);
    for i in 0..config.num_layers {
        let layer_prefix = format!("{prefix}layers.{i}.");
        let layer = load_layer(&tensor_map, &layer_prefix, i, config)?;
        layers.push(layer);
        if let Some(pb) = &pb {
            pb.inc(1);
        }
    }
    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    // Load final norm.
    let final_norm = extract_tensor(&tensor_map, &format!("{prefix}norm.weight"))?;
    validate_shape(&final_norm, &[config.hidden_size], "final_norm")?;

    // Optional: load native MTP head when the checkpoint ships it.
    // Qwen3.5-4B has `num_nextn_predict_layers = 1` (k=1 draft depth) and
    // stores 15 `mtp.*` tensors. We detect purely by tensor presence rather
    // than adding a new config field — if the checkpoint has MTP tensors,
    // load them; if not, leave `ModelWeights.mtp` as None.
    let mtp = if options.load_mtp {
        load_mtp_if_present(&tensor_map, &prefix, config)?
    } else {
        None
    };
    let deferred_mtp = if options.load_mtp {
        None
    } else {
        detect_mtp_prefix(&tensor_map, &prefix).map(|mtp_prefix| DeferredMtpSource {
            model_dir: model_dir.to_path_buf(),
            mtp_prefix,
            config: config.clone(),
            _snapshot_lease: Arc::clone(&snapshot_lease),
        })
    };
    if mtp.is_some() {
        tracing::info!("Native MTP head detected and loaded (k=1 draft depth)");
    } else if let Some(source) = &deferred_mtp {
        tracing::info!(
            mtp_prefix = %source.mtp_prefix,
            "Native MTP head detected; deferring CPU load until first use"
        );
    } else if !options.load_mtp {
        tracing::info!("Skipping native MTP head load");
    }

    let weights = ModelWeights {
        source_content_sha256: Some(source_content_sha256),
        source_content_guard: Some(source_content_guard),
        embedding: EmbeddingWeights { embed_tokens },
        layers,
        final_norm,
        mtp,
        deferred_mtp,
    };

    let total_mb = weights.total_bytes() as f64 / (1024.0 * 1024.0);
    let total_params_m = weights.total_params() as f64 / 1_000_000.0;
    tracing::info!(
        "Loaded {:.0}M parameters ({:.0} MB) across {} layers",
        total_params_m,
        total_mb,
        config.num_layers,
    );

    Ok(weights)
}

/// Load GPTQ-quantized model weights, dequantizing INT4 to BF16.
///
/// GPTQ models store linear layer weights as packed INT4 with per-group scales
/// and zero points. This function loads and dequantizes them to BF16 `WeightTensor`s
/// that fit into the standard `ModelWeights` structure.
///
/// Non-linear weights (embeddings, layer norms, conv1d, etc.) are loaded as-is
/// since they are stored in their original precision even in GPTQ models.
fn load_model_gptq(
    model_dir: &Path,
    config: &ModelConfig,
    gptq_config: &GptqConfig,
    _options: LoadModelOptions,
    snapshot_lease: Arc<ModelSnapshotLease>,
) -> Result<ModelWeights> {
    let shards = discover_shards(model_dir)?;
    tracing::info!(
        "Loading GPTQ model from {} ({} shard{})",
        model_dir.display(),
        shards.len(),
        if shards.len() == 1 { "" } else { "s" }
    );

    let loaded_shards = mmap_shards(&shards)?;
    let source_content_guard = loaded_shard_content_guard(&loaded_shards, snapshot_lease);
    let source_content_sha256 = source_content_guard.initial_sha256().to_string();
    let parsed: Vec<SafeTensors<'_>> = loaded_shards
        .iter()
        .map(|shard| {
            SafeTensors::deserialize(&shard.mmap[..]).context("Failed to parse safetensors file")
        })
        .collect::<Result<Vec<_>>>()?;

    let tensor_map = build_tensor_map(&parsed, &loaded_shards);

    let prefix = detect_prefix(&tensor_map);
    tracing::info!("Using weight name prefix: \"{prefix}\"");

    // Embedding — stored in original precision even in GPTQ models.
    let embed_tokens = extract_tensor(&tensor_map, &format!("{prefix}embed_tokens.weight"))?;
    validate_shape(
        &embed_tokens,
        &[config.vocab_size, config.hidden_size],
        "embed_tokens",
    )?;

    // Load layers — linear projections are GPTQ-quantized, norms are dense.
    let mut layers = Vec::with_capacity(config.num_layers);
    let pb = make_layer_progress(config.num_layers);
    for i in 0..config.num_layers {
        let layer_prefix = format!("{prefix}layers.{i}.");
        let layer = load_layer_gptq(&tensor_map, &layer_prefix, i, config, gptq_config)?;
        layers.push(layer);
        if let Some(pb) = &pb {
            pb.inc(1);
        }
    }
    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    // Final norm — stored in original precision.
    let final_norm = extract_tensor(&tensor_map, &format!("{prefix}norm.weight"))?;
    validate_shape(&final_norm, &[config.hidden_size], "final_norm")?;

    // GPTQ MTP is not yet supported — the MTP layer projections would need
    // a separate GPTQ dequant path. For now we simply log and skip.
    if tensor_map.contains_key("mtp.fc.weight")
        || tensor_map.contains_key(format!("{prefix}mtp.fc.weight").as_str())
    {
        tracing::warn!(
            "MTP tensors present in GPTQ checkpoint but GPTQ MTP loading is not yet implemented — skipping"
        );
    }

    let weights = ModelWeights {
        source_content_sha256: Some(source_content_sha256),
        source_content_guard: Some(source_content_guard),
        embedding: EmbeddingWeights { embed_tokens },
        layers,
        final_norm,
        mtp: None,
        deferred_mtp: None,
    };

    let total_mb = weights.total_bytes() as f64 / (1024.0 * 1024.0);
    let total_params_m = weights.total_params() as f64 / 1_000_000.0;
    tracing::info!(
        "Loaded {:.0}M parameters ({:.0} MB dequantized BF16) across {} layers (GPTQ INT{})",
        total_params_m,
        total_mb,
        config.num_layers,
        gptq_config.bits,
    );

    Ok(weights)
}

fn create_immutable_model_snapshot(
    model_dir: &Path,
    configured_snapshot_root: Option<&Path>,
) -> Result<ImmutableModelSnapshot> {
    let source_root = model_dir
        .canonicalize()
        .with_context(|| format!("resolve model directory {}", model_dir.display()))?;
    ensure_directory(&source_root, "model directory")?;
    let shards = discover_shards(&source_root)?;

    let mut inputs = shards
        .into_iter()
        .map(|path| {
            let filename = path
                .file_name()
                .context("model shard path has no filename")?
                .to_owned();
            Ok((path, PathBuf::from(filename)))
        })
        .collect::<Result<Vec<_>>>()?;
    let gptq_config = source_root.join(GPTQ_CONFIG_FILENAME);
    match fs::symlink_metadata(&gptq_config) {
        Ok(metadata) => {
            ensure_regular_metadata(&gptq_config, &metadata, "GPTQ config")?;
            inputs.push((gptq_config, PathBuf::from(GPTQ_CONFIG_FILENAME)));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("inspect {}", gptq_config.display()));
        }
    }

    let total_bytes = inputs.iter().try_fold(0u64, |total, (path, _)| {
        let metadata = fs::symlink_metadata(path)
            .with_context(|| format!("inspect snapshot input {}", path.display()))?;
        ensure_regular_metadata(path, &metadata, "model snapshot input")?;
        total
            .checked_add(metadata.len())
            .context("model snapshot byte count overflow")
    })?;
    if total_bytes > MAX_SNAPSHOT_INPUT_BYTES {
        bail!(
            "model snapshot inputs require {total_bytes} bytes, exceeding the {MAX_SNAPSHOT_INPUT_BYTES}-byte safety limit"
        );
    }

    let directory = create_snapshot_directory(&source_root, configured_snapshot_root)?;
    set_private_directory_permissions(directory.path())?;
    let mut remaining_bytes = total_bytes;
    let mut cloned_files = 0usize;
    let mut copied_files = 0usize;
    for (source, relative) in &inputs {
        let destination = directory.path().join(relative);
        let copied = copy_snapshot_input(source, &destination, remaining_bytes, directory.path())?;
        if copied {
            copied_files += 1;
        } else {
            cloned_files += 1;
        }
        remaining_bytes = remaining_bytes.saturating_sub(
            fs::metadata(&destination)
                .with_context(|| format!("stat copied snapshot input {}", destination.display()))?
                .len(),
        );
    }
    #[cfg(unix)]
    File::open(directory.path())
        .and_then(|file| file.sync_all())
        .with_context(|| {
            format!(
                "sync model snapshot directory {}",
                directory.path().display()
            )
        })?;

    tracing::info!(
        source = %source_root.display(),
        snapshot = %directory.path().display(),
        files = inputs.len(),
        logical_bytes = total_bytes,
        cloned_files,
        copied_files,
        "created private immutable model snapshot"
    );
    Ok(ImmutableModelSnapshot {
        lease: Arc::new(ModelSnapshotLease::new(directory)),
    })
}

fn create_snapshot_directory(
    source_root: &Path,
    configured_snapshot_root: Option<&Path>,
) -> Result<tempfile::TempDir> {
    let builder = || {
        let mut builder = tempfile::Builder::new();
        builder.prefix(".kiln-model-snapshot-");
        builder
    };
    if let Some(root) = configured_snapshot_root {
        fs::create_dir_all(&root)
            .with_context(|| format!("create configured snapshot root {}", root.display()))?;
        ensure_directory(&root, "configured model snapshot root")?;
        return builder()
            .tempdir_in(&root)
            .with_context(|| format!("create private snapshot under {}", root.display()));
    }

    if let Some(parent) = source_root.parent() {
        match builder().tempdir_in(parent) {
            Ok(directory) => return Ok(directory),
            Err(error) => tracing::warn!(
                parent = %parent.display(),
                error = %error,
                "could not create model snapshot beside source; falling back to system temp"
            ),
        }
    }
    builder()
        .tempdir()
        .context("create private model snapshot in system temp")
}

fn copy_snapshot_input(
    source_path: &Path,
    destination_path: &Path,
    remaining_logical_bytes: u64,
    snapshot_root: &Path,
) -> Result<bool> {
    let mut source = open_regular_source(source_path, "model snapshot input")?;
    let expected_len = source
        .metadata()
        .with_context(|| format!("stat opened snapshot input {}", source_path.display()))?
        .len();
    let (mut destination, cloned) =
        create_or_clone_destination(source_path, &source, destination_path)?;

    if !cloned {
        ensure_snapshot_copy_capacity(snapshot_root, remaining_logical_bytes)?;
        source
            .seek(SeekFrom::Start(0))
            .with_context(|| format!("seek snapshot input {}", source_path.display()))?;
        destination
            .seek(SeekFrom::Start(0))
            .with_context(|| format!("seek snapshot destination {}", destination_path.display()))?;
        let mut remaining = expected_len;
        let mut buffer = vec![0u8; SNAPSHOT_COPY_BUFFER_BYTES];
        while remaining > 0 {
            let request = usize::try_from(remaining.min(buffer.len() as u64))
                .expect("bounded snapshot read length fits usize");
            let read = source
                .read(&mut buffer[..request])
                .with_context(|| format!("read snapshot input {}", source_path.display()))?;
            if read == 0 {
                bail!(
                    "model snapshot input {} became shorter while copied: expected {expected_len} bytes",
                    source_path.display()
                );
            }
            destination.write_all(&buffer[..read]).map_err(|error| {
                snapshot_write_error(error, destination_path, remaining_logical_bytes)
            })?;
            remaining -= read as u64;
        }
        let mut extra = [0u8; 1];
        if source
            .read(&mut extra)
            .with_context(|| format!("check snapshot input length {}", source_path.display()))?
            != 0
        {
            bail!(
                "model snapshot input {} grew while copied beyond {expected_len} bytes",
                source_path.display()
            );
        }
    }

    destination
        .sync_all()
        .map_err(|error| snapshot_write_error(error, destination_path, remaining_logical_bytes))?;
    let observed_len = destination
        .metadata()
        .with_context(|| format!("stat snapshot destination {}", destination_path.display()))?
        .len();
    if observed_len != expected_len {
        bail!(
            "model snapshot destination {} has {observed_len} bytes, expected {expected_len}",
            destination_path.display()
        );
    }
    set_read_only_file_permissions(destination_path)?;
    Ok(!cloned)
}

fn snapshot_write_error(error: io::Error, path: &Path, required: u64) -> anyhow::Error {
    let storage_full = error.kind() == io::ErrorKind::StorageFull
        || matches!(error.raw_os_error(), Some(28) | Some(112));
    if storage_full {
        anyhow::anyhow!(
            "model snapshot filesystem ran out of space while writing {} (up to {required} additional bytes required); configure the model snapshot directory on a filesystem with sufficient capacity: {error}",
            path.display()
        )
    } else {
        anyhow::anyhow!("write model snapshot file {}: {error}", path.display())
    }
}

fn ensure_snapshot_copy_capacity(path: &Path, required: u64) -> Result<()> {
    let Some(available) = available_space(path)? else {
        return Ok(());
    };
    let reserve = SNAPSHOT_FREE_SPACE_RESERVE_BYTES.max(required / 100);
    let required_with_reserve = required.saturating_add(reserve);
    if available < required_with_reserve {
        bail!(
            "model snapshot copy requires {required} bytes plus {reserve} bytes reserve, but only {available} bytes are available on {}; configure the model snapshot directory on a filesystem with sufficient capacity",
            path.display()
        );
    }
    Ok(())
}

#[cfg(unix)]
fn available_space(path: &Path) -> Result<Option<u64>> {
    use std::os::unix::ffi::OsStrExt;
    let path = CString::new(path.as_os_str().as_bytes())
        .context("model snapshot path contains a NUL byte")?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `path` is NUL terminated and `stat` points to writable storage.
    if unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error()).context("inspect snapshot filesystem capacity");
    }
    // SAFETY: successful `statvfs` initialized the structure.
    let stat = unsafe { stat.assume_init() };
    let fragment_size = if stat.f_frsize == 0 {
        stat.f_bsize
    } else {
        stat.f_frsize
    };
    Ok(Some(
        (stat.f_bavail as u64).saturating_mul(fragment_size as u64),
    ))
}

#[cfg(not(unix))]
fn available_space(_path: &Path) -> Result<Option<u64>> {
    Ok(None)
}

fn create_or_clone_destination(
    _source_path: &Path,
    source: &File,
    destination_path: &Path,
) -> Result<(File, bool)> {
    #[cfg(target_os = "macos")]
    if try_macos_clone(_source_path, destination_path)? {
        let destination = OpenOptions::new()
            .read(true)
            .write(true)
            .open(destination_path)
            .with_context(|| format!("open cloned snapshot {}", destination_path.display()))?;
        return Ok((destination, true));
    }

    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    let destination = options
        .open(destination_path)
        .with_context(|| format!("create snapshot destination {}", destination_path.display()))?;

    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: both descriptors are live regular files. FICLONE either
        // creates an independent CoW extent mapping or leaves the destination
        // available for the explicit-copy fallback below.
        let result = unsafe {
            libc::ioctl(
                destination.as_raw_fd(),
                libc::FICLONE as _,
                source.as_raw_fd(),
            )
        };
        if result == 0 {
            return Ok((destination, true));
        }
        let error = io::Error::last_os_error();
        if !matches!(
            error.raw_os_error(),
            Some(libc::EXDEV)
                | Some(libc::EOPNOTSUPP)
                | Some(libc::ENOTTY)
                | Some(libc::EINVAL)
                | Some(libc::EPERM)
                | Some(libc::ENOSYS)
        ) {
            return Err(error).with_context(|| {
                format!("reflink model snapshot to {}", destination_path.display())
            });
        }
        destination
            .set_len(0)
            .with_context(|| format!("reset reflink fallback {}", destination_path.display()))?;
    }

    Ok((destination, false))
}

#[cfg(target_os = "macos")]
fn try_macos_clone(source: &Path, destination: &Path) -> Result<bool> {
    use std::os::unix::ffi::OsStrExt;
    let source = CString::new(source.as_os_str().as_bytes())
        .context("model source path contains a NUL byte")?;
    let destination = CString::new(destination.as_os_str().as_bytes())
        .context("model snapshot path contains a NUL byte")?;
    // SAFETY: both C strings are live for this call; destination does not yet
    // exist and resides inside the private snapshot directory.
    if unsafe { libc::clonefile(source.as_ptr(), destination.as_ptr(), 0) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if matches!(
        error.raw_os_error(),
        Some(libc::EXDEV)
            | Some(libc::ENOTSUP)
            | Some(libc::EINVAL)
            | Some(libc::EPERM)
            | Some(libc::ENOSYS)
    ) {
        Ok(false)
    } else {
        Err(error).with_context(|| "clone model snapshot with clonefile")
    }
}

fn set_private_directory_permissions(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("set private permissions on {}", path.display()))?;
    }
    Ok(())
}

fn set_read_only_file_permissions(path: &Path) -> Result<()> {
    #[cfg(unix)]
    let permissions = {
        use std::os::unix::fs::PermissionsExt;
        fs::Permissions::from_mode(0o400)
    };
    #[cfg(not(unix))]
    let permissions = {
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_readonly(true);
        permissions
    };
    fs::set_permissions(path, permissions)
        .with_context(|| format!("make snapshot input read-only {}", path.display()))
}

fn ensure_directory(path: &Path, label: &str) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("inspect {label} {}", path.display()))?;
    if !metadata.is_dir() {
        bail!("{label} is not a directory: {}", path.display());
    }
    Ok(())
}

fn ensure_regular_metadata(path: &Path, metadata: &fs::Metadata, label: &str) -> Result<()> {
    if metadata.file_type().is_symlink() {
        bail!("{label} must not be a symlink: {}", path.display());
    }
    if !metadata.is_file() {
        bail!("{label} is not a regular file: {}", path.display());
    }
    Ok(())
}

fn open_regular_source(path: &Path, label: &str) -> Result<File> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspect {label} {}", path.display()))?;
    ensure_regular_metadata(path, &metadata, label)?;
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    let file = options
        .open(path)
        .with_context(|| format!("open {label} {}", path.display()))?;
    let opened = file
        .metadata()
        .with_context(|| format!("stat opened {label} {}", path.display()))?;
    if !opened.is_file() {
        bail!("opened {label} is not a regular file: {}", path.display());
    }
    Ok(file)
}

fn read_bounded_regular_file(path: &Path, label: &str, max_bytes: u64) -> Result<Vec<u8>> {
    let file = open_regular_source(path, label)?;
    let length = file.metadata()?.len();
    if length > max_bytes {
        bail!(
            "{label} {} is {length} bytes, exceeding the {max_bytes}-byte safety limit",
            path.display()
        );
    }
    let capacity = usize::try_from(length).context("bounded file length does not fit usize")?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("read {label} {}", path.display()))?;
    if bytes.len() as u64 > max_bytes {
        bail!("{label} {} grew beyond the safety limit", path.display());
    }
    Ok(bytes)
}

fn validate_shard_filename(filename: &str) -> Result<&str> {
    let path = Path::new(filename);
    let mut components = path.components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => {}
        _ => bail!(
            "safetensors index shard path must be one relative filename without traversal: {filename:?}"
        ),
    }
    if filename.as_bytes().len() > 255 || !filename.ends_with(".safetensors") {
        bail!("invalid safetensors shard filename in index: {filename:?}");
    }
    Ok(filename)
}

/// Discover a bounded, path-safe set of regular safetensors shard files.
fn discover_shards(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = model_dir.join(INDEX_FILENAME);
    let has_index = match fs::symlink_metadata(&index_path) {
        Ok(_) => true,
        Err(error) if error.kind() == io::ErrorKind::NotFound => false,
        Err(error) => {
            return Err(error)
                .with_context(|| format!("inspect safetensors index {}", index_path.display()));
        }
    };

    if has_index {
        // Sharded model: read index to find shard filenames.
        let index_bytes =
            read_bounded_regular_file(&index_path, "safetensors index", MAX_INDEX_BYTES)?;
        let index: serde_json::Value = serde_json::from_slice(&index_bytes)
            .context("Failed to parse safetensors index JSON")?;

        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .context("Index file missing 'weight_map' object")?;

        // Collect unique shard filenames, preserving order.
        let mut seen = HashSet::new();
        let mut shard_files = Vec::new();
        for filename in weight_map.values() {
            let filename = filename
                .as_str()
                .context("weight_map value is not a string")?;
            let filename = validate_shard_filename(filename)?;
            if seen.insert(filename.to_string()) {
                let shard_path = model_dir.join(filename);
                let metadata = fs::symlink_metadata(&shard_path).with_context(|| {
                    format!("inspect shard referenced by index {}", shard_path.display())
                })?;
                ensure_regular_metadata(&shard_path, &metadata, "model shard")?;
                shard_files.push(shard_path);
                if shard_files.len() > MAX_SHARD_COUNT {
                    bail!("model index exceeds the {MAX_SHARD_COUNT}-shard safety limit");
                }
            }
        }

        if shard_files.is_empty() {
            bail!("Index file has an empty weight_map");
        }

        shard_files.sort();
        Ok(shard_files)
    } else {
        // Single-file model.
        let single = model_dir.join("model.safetensors");
        match fs::symlink_metadata(&single) {
            Ok(metadata) => {
                ensure_regular_metadata(&single, &metadata, "model shard")?;
                Ok(vec![single])
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                // Try glob for sharded files without index.
                let mut shards = Vec::new();
                for entry in fs::read_dir(model_dir)
                    .with_context(|| format!("enumerate model directory {}", model_dir.display()))?
                {
                    let entry = entry.with_context(|| {
                        format!("enumerate model directory {}", model_dir.display())
                    })?;
                    let path = entry.path();
                    if path.extension().is_some_and(|ext| ext == "safetensors") {
                        let metadata = fs::symlink_metadata(&path)
                            .with_context(|| format!("inspect model shard {}", path.display()))?;
                        ensure_regular_metadata(&path, &metadata, "model shard")?;
                        shards.push(path);
                        if shards.len() > MAX_SHARD_COUNT {
                            bail!(
                                "model directory exceeds the {MAX_SHARD_COUNT}-shard safety limit"
                            );
                        }
                    }
                }
                shards.sort();

                if shards.is_empty() {
                    bail!("No .safetensors files found in {}", model_dir.display());
                }
                Ok(shards)
            }
            Err(error) => {
                Err(error).with_context(|| format!("inspect model shard {}", single.display()))
            }
        }
    }
}

/// Memory-map all shard files.
fn mmap_shards(paths: &[std::path::PathBuf]) -> Result<Vec<LoadedShard>> {
    paths
        .iter()
        .map(|path| {
            let file = Arc::new(open_regular_source(path, "model snapshot shard")?);
            let meta = file
                .metadata()
                .with_context(|| format!("Failed to stat {}", path.display()))?;
            let modified_ns = meta
                .modified()
                .ok()
                .and_then(|mtime| mtime.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let shard_meta = Arc::new(ShardMetadata {
                path: path.clone(),
                size_bytes: meta.len(),
                modified_ns,
            });
            // SAFETY: We assume the file is not modified while mapped.
            // This is standard practice for model weight loading.
            let mmap = unsafe { Mmap::map(file.as_ref()) }
                .with_context(|| format!("Failed to mmap {}", path.display()))?;
            Ok(LoadedShard {
                meta: shard_meta,
                file,
                mmap: Arc::new(mmap),
            })
        })
        .collect()
}

fn loaded_shard_content_guard(
    shards: &[LoadedShard],
    snapshot_lease: Arc<ModelSnapshotLease>,
) -> SourceContentGuard {
    SourceContentGuard::new(
        shards
            .iter()
            .map(|shard| (Arc::clone(&shard.file), Arc::clone(&shard.mmap)))
            .collect(),
        snapshot_lease,
    )
}

#[cfg(test)]
fn loaded_shard_content_sha256(shards: &[LoadedShard]) -> String {
    let directory = tempfile::tempdir().unwrap();
    let lease = Arc::new(ModelSnapshotLease::new(directory));
    loaded_shard_content_guard(shards, lease)
        .initial_sha256()
        .to_string()
}

/// Detect the weight name prefix by checking which keys exist.
fn detect_prefix(tensor_map: &TensorMap<'_>) -> String {
    // Try the VL model prefix first (Qwen3.5-4B ships as VL).
    if tensor_map.contains_key(format!("{LM_PREFIX}embed_tokens.weight").as_str()) {
        return LM_PREFIX.to_string();
    }
    // Fall back to bare prefix for text-only checkpoints.
    if tensor_map.contains_key("model.embed_tokens.weight") {
        return "model.".to_string();
    }
    // Last resort: no prefix.
    if tensor_map.contains_key("embed_tokens.weight") {
        return String::new();
    }
    // Default to VL prefix and let it fail with a clear error.
    LM_PREFIX.to_string()
}

/// Load one transformer layer's weights.
fn load_layer(
    tensor_map: &TensorMap<'_>,
    prefix: &str,
    layer_idx: usize,
    config: &ModelConfig,
) -> Result<LayerWeights> {
    let input_layernorm = extract_tensor(tensor_map, &format!("{prefix}input_layernorm.weight"))?;
    validate_shape(
        &input_layernorm,
        &[config.hidden_size],
        &format!("layer {layer_idx} input_layernorm"),
    )?;

    let post_attention_layernorm = extract_tensor(
        tensor_map,
        &format!("{prefix}post_attention_layernorm.weight"),
    )?;
    validate_shape(
        &post_attention_layernorm,
        &[config.hidden_size],
        &format!("layer {layer_idx} post_attn_layernorm"),
    )?;

    let mlp = load_ffn(tensor_map, prefix, layer_idx, config)?;

    let attention = if config.is_full_attention_layer(layer_idx) {
        AttentionWeights::Full(load_full_attention(tensor_map, prefix, layer_idx, config)?)
    } else {
        AttentionWeights::Linear(load_linear_attention(
            tensor_map, prefix, layer_idx, config,
        )?)
    };

    Ok(LayerWeights {
        input_layernorm,
        post_attention_layernorm,
        attention,
        mlp,
    })
}

/// Load full GQA attention weights.
fn load_full_attention(
    tensor_map: &TensorMap<'_>,
    prefix: &str,
    layer_idx: usize,
    config: &ModelConfig,
) -> Result<FullAttentionWeights> {
    let attn = format!("{prefix}self_attn.");
    let ctx = |name: &str| format!("layer {layer_idx} self_attn.{name}");

    let q_proj_dim = config.full_attn_q_proj_dim();
    let q_out_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    let q_proj = extract_tensor(tensor_map, &format!("{attn}q_proj.weight"))?;
    validate_shape(&q_proj, &[q_proj_dim, config.hidden_size], &ctx("q_proj"))?;

    let k_proj = extract_tensor(tensor_map, &format!("{attn}k_proj.weight"))?;
    validate_shape(&k_proj, &[kv_dim, config.hidden_size], &ctx("k_proj"))?;

    let v_proj = extract_tensor(tensor_map, &format!("{attn}v_proj.weight"))?;
    validate_shape(&v_proj, &[kv_dim, config.hidden_size], &ctx("v_proj"))?;

    let o_proj = extract_tensor(tensor_map, &format!("{attn}o_proj.weight"))?;
    validate_shape(&o_proj, &[config.hidden_size, q_out_dim], &ctx("o_proj"))?;

    let q_norm = extract_tensor(tensor_map, &format!("{attn}q_norm.weight"))?;
    validate_shape(&q_norm, &[config.head_dim], &ctx("q_norm"))?;

    let k_norm = extract_tensor(tensor_map, &format!("{attn}k_norm.weight"))?;
    validate_shape(&k_norm, &[config.head_dim], &ctx("k_norm"))?;

    Ok(FullAttentionWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
    })
}

/// Load Gated DeltaNet linear attention weights.
fn load_linear_attention(
    tensor_map: &TensorMap<'_>,
    prefix: &str,
    layer_idx: usize,
    config: &ModelConfig,
) -> Result<LinearAttentionWeights> {
    let attn = format!("{prefix}linear_attn.");
    let ctx = |name: &str| format!("layer {layer_idx} linear_attn.{name}");

    // Linear attention uses asymmetric Q/K/V dimensions from config.
    let fused_qkv_dim = config.linear_qkv_dim(); // Q + K + V total
    let v_dim = config.linear_v_dim();
    let num_heads = config.linear_num_value_heads;

    let in_proj_qkv = extract_tensor(tensor_map, &format!("{attn}in_proj_qkv.weight"))?;
    validate_shape(
        &in_proj_qkv,
        &[fused_qkv_dim, config.hidden_size],
        &ctx("in_proj_qkv"),
    )?;

    let in_proj_z = extract_tensor(tensor_map, &format!("{attn}in_proj_z.weight"))?;
    validate_shape(&in_proj_z, &[v_dim, config.hidden_size], &ctx("in_proj_z"))?;

    let out_proj = extract_tensor(tensor_map, &format!("{attn}out_proj.weight"))?;
    validate_shape(&out_proj, &[config.hidden_size, v_dim], &ctx("out_proj"))?;

    let in_proj_a = extract_tensor(tensor_map, &format!("{attn}in_proj_a.weight"))?;
    validate_shape(
        &in_proj_a,
        &[num_heads, config.hidden_size],
        &ctx("in_proj_a"),
    )?;

    let in_proj_b = extract_tensor(tensor_map, &format!("{attn}in_proj_b.weight"))?;
    validate_shape(
        &in_proj_b,
        &[num_heads, config.hidden_size],
        &ctx("in_proj_b"),
    )?;

    // conv1d shape is [channels, 1, kernel_size] — channels = fused QKV dim.
    let conv1d = extract_tensor(tensor_map, &format!("{attn}conv1d.weight"))?;
    if conv1d.shape.len() != 3 || conv1d.shape[0] != fused_qkv_dim || conv1d.shape[1] != 1 {
        bail!(
            "Shape mismatch for {}: expected [{fused_qkv_dim}, 1, *], got {:?}",
            ctx("conv1d"),
            conv1d.shape
        );
    }

    let norm = extract_tensor(tensor_map, &format!("{attn}norm.weight"))?;
    // Group norm weight — per-head normalization.
    validate_shape(&norm, &[config.linear_key_head_dim], &ctx("norm"))?;

    let a_log = extract_tensor(tensor_map, &format!("{attn}A_log"))?;
    if a_log.numel() != num_heads {
        bail!(
            "Element count mismatch for {}: expected {num_heads}, got {} (shape {:?})",
            ctx("A_log"),
            a_log.numel(),
            a_log.shape
        );
    }

    let dt_bias = extract_tensor(tensor_map, &format!("{attn}dt_bias"))?;
    if dt_bias.numel() != num_heads {
        bail!(
            "Element count mismatch for {}: expected {num_heads}, got {} (shape {:?})",
            ctx("dt_bias"),
            dt_bias.numel(),
            dt_bias.shape
        );
    }

    Ok(LinearAttentionWeights {
        in_proj_qkv,
        in_proj_z,
        out_proj,
        in_proj_a,
        in_proj_b,
        conv1d,
        norm,
        a_log,
        dt_bias,
    })
}

/// Load FFN/MLP weights.
fn load_ffn(
    tensor_map: &TensorMap<'_>,
    prefix: &str,
    layer_idx: usize,
    config: &ModelConfig,
) -> Result<FfnWeights> {
    let mlp = format!("{prefix}mlp.");
    let ctx = |name: &str| format!("layer {layer_idx} mlp.{name}");

    let gate_proj = extract_tensor(tensor_map, &format!("{mlp}gate_proj.weight"))?;
    validate_shape(
        &gate_proj,
        &[config.intermediate_size, config.hidden_size],
        &ctx("gate_proj"),
    )?;

    let up_proj = extract_tensor(tensor_map, &format!("{mlp}up_proj.weight"))?;
    validate_shape(
        &up_proj,
        &[config.intermediate_size, config.hidden_size],
        &ctx("up_proj"),
    )?;

    let down_proj = extract_tensor(tensor_map, &format!("{mlp}down_proj.weight"))?;
    validate_shape(
        &down_proj,
        &[config.hidden_size, config.intermediate_size],
        &ctx("down_proj"),
    )?;

    Ok(FfnWeights {
        gate_proj,
        up_proj,
        down_proj,
    })
}

/// Detect which MTP prefix (if any) the loaded checkpoint uses.
///
/// Qwen3.5-4B publishes MTP tensors under the bare `mtp.` prefix. A future
/// VL-wrapped checkpoint could instead nest them under the language-model
/// prefix (e.g. `model.language_model.mtp.`). Probe both by checking for
/// `{candidate}fc.weight`, which is always present when MTP is shipped.
/// Returns `None` if no MTP tensors are present (checkpoint is MTP-less).
fn detect_mtp_prefix(tensor_map: &TensorMap<'_>, base_prefix: &str) -> Option<String> {
    let prefixed = format!("{base_prefix}mtp.");
    if tensor_map.contains_key(format!("{prefixed}fc.weight").as_str()) {
        return Some(prefixed);
    }
    if tensor_map.contains_key("mtp.fc.weight") {
        return Some("mtp.".to_string());
    }
    None
}

/// Detect and load the native MTP head if present in the checkpoint.
///
/// Qwen3.5-4B ships 15 MTP-prefixed tensors. The top-level prefix can be
/// either the VL language-model prefix (`{prefix}mtp.*`, e.g. if a future
/// checkpoint re-exports the VL wrapper) or bare `mtp.*` (the layout that
/// `Qwen/Qwen3.5-4B` actually publishes on the Hub). Similarly, the final
/// RMSNorm key is `mtp.norm.weight` in the published checkpoint but older
/// docs / vLLM references call it `mtp.final_layernorm.weight`. Detect
/// both layouts so `KILN_SPEC_METHOD=mtp` works on the stock release.
///
/// Tensors loaded (Qwen3.5-4B layout):
/// - `mtp.fc.weight` `[hidden, 2*hidden]`
/// - `mtp.pre_fc_norm_embedding.weight` `[hidden]`
/// - `mtp.pre_fc_norm_hidden.weight` `[hidden]`
/// - `mtp.layers.0.*` — one full GQA transformer layer (shape identical to a
///   main-model full-attention layer including `attn_output_gate`)
/// - `mtp.norm.weight` (or `mtp.final_layernorm.weight`) `[hidden]`
///
/// The MTP head ties its `lm_head` to the base model's `embed_tokens`, so we
/// do NOT load a separate `mtp.lm_head` tensor. The spec-decode forward
/// path reuses `GpuWeights::embed_tokens_t`.
///
/// Returns `Ok(Some(..))` when detected, `Ok(None)` when absent (so older
/// or MTP-less checkpoints continue to load unchanged).
fn load_mtp_if_present(
    tensor_map: &TensorMap<'_>,
    prefix: &str,
    config: &ModelConfig,
) -> Result<Option<MtpWeights>> {
    // Probe: is the MTP fc projection present under either the VL prefix
    // or the bare `mtp.` prefix? Qwen3.5-4B publishes bare; keep the
    // VL-prefixed form working for any re-exports that embed the LM inside
    // the VL wrapper tensors.
    let mtp_prefix = match detect_mtp_prefix(tensor_map, prefix) {
        Some(p) => p,
        None => return Ok(None),
    };
    load_mtp_with_prefix(tensor_map, &mtp_prefix, config).map(Some)
}

fn load_mtp_with_prefix(
    tensor_map: &TensorMap<'_>,
    mtp_prefix: &str,
    config: &ModelConfig,
) -> Result<MtpWeights> {
    let ctx = |name: &str| format!("{mtp_prefix}{name}");

    let fc = extract_tensor(tensor_map, &format!("{mtp_prefix}fc.weight"))?;
    // fc maps concat(embed, hidden) → hidden, so shape is [hidden, 2*hidden].
    validate_shape(
        &fc,
        &[config.hidden_size, 2 * config.hidden_size],
        &ctx("fc"),
    )?;

    let pre_fc_norm_embedding = extract_tensor(
        tensor_map,
        &format!("{mtp_prefix}pre_fc_norm_embedding.weight"),
    )?;
    validate_shape(
        &pre_fc_norm_embedding,
        &[config.hidden_size],
        &ctx("pre_fc_norm_embedding"),
    )?;

    let pre_fc_norm_hidden = extract_tensor(
        tensor_map,
        &format!("{mtp_prefix}pre_fc_norm_hidden.weight"),
    )?;
    validate_shape(
        &pre_fc_norm_hidden,
        &[config.hidden_size],
        &ctx("pre_fc_norm_hidden"),
    )?;

    // Stock Qwen3.5-4B names this `mtp.norm.weight`; older vLLM / HF docs
    // (and earlier kiln comments) refer to it as `mtp.final_layernorm`.
    // Accept either.
    let final_ln_key = [
        format!("{mtp_prefix}norm.weight"),
        format!("{mtp_prefix}final_layernorm.weight"),
    ]
    .into_iter()
    .find(|k| tensor_map.contains_key(k.as_str()))
    .with_context(|| {
        format!(
            "MTP final RMSNorm not found (looked for {mtp_prefix}norm.weight and \
             {mtp_prefix}final_layernorm.weight)"
        )
    })?;
    let final_layernorm = extract_tensor(tensor_map, &final_ln_key)?;
    validate_shape(
        &final_layernorm,
        &[config.hidden_size],
        &ctx("final_layernorm"),
    )?;

    // The MTP layer uses the same full-attention shape as the main-model
    // full-attention layers (GQA + output gate + SwiGLU MLP). We reuse
    // `load_layer` with a synthetic prefix that points at `mtp.layers.0.`
    // and an index (3) whose residue class `(i + 1) % 4 == 0` makes
    // `is_full_attention_layer` return true. The layer_idx argument is only
    // used for error context strings, not for dispatch.
    let mtp_layer_prefix = format!("{mtp_prefix}layers.0.");
    let layer = load_layer(tensor_map, &mtp_layer_prefix, 3, config).context("mtp layer 0")?;
    // Defensive: MTP layer must be full attention. If this ever fires the
    // checkpoint is shipping a linear-attention MTP head and the forward
    // pass below won't match.
    match &layer.attention {
        AttentionWeights::Full(_) => {}
        AttentionWeights::Linear(_) => {
            bail!(
                "MTP layer loaded as linear attention — expected full GQA attention. \
                 Checkpoint schema change?"
            );
        }
    }

    Ok(MtpWeights {
        fc,
        pre_fc_norm_embedding,
        pre_fc_norm_hidden,
        layer,
        final_layernorm,
    })
}

/// Load only the checkpoint's native-MTP tensors from a previously recorded
/// deferred source.
pub fn load_deferred_mtp(source: &DeferredMtpSource) -> Result<MtpWeights> {
    let shards = discover_shards(&source.model_dir)?;
    let loaded_shards = mmap_shards(&shards)?;
    let parsed: Vec<SafeTensors<'_>> = loaded_shards
        .iter()
        .map(|shard| {
            SafeTensors::deserialize(&shard.mmap[..]).context("Failed to parse safetensors file")
        })
        .collect::<Result<Vec<_>>>()?;

    let tensor_map = build_tensor_map(&parsed, &loaded_shards);

    load_mtp_with_prefix(&tensor_map, &source.mtp_prefix, &source.config).with_context(|| {
        format!(
            "deferred native MTP load from {}",
            source.model_dir.display()
        )
    })
}

fn build_tensor_map<'data>(
    parsed: &[SafeTensors<'data>],
    loaded_shards: &[LoadedShard],
) -> TensorMap<'data> {
    let mut tensor_map: TensorMap<'data> = HashMap::new();
    for (shard_idx, st) in parsed.iter().enumerate() {
        for (name, view) in st.tensors() {
            tensor_map.insert(
                name,
                TensorMapEntry {
                    shard: Arc::clone(&loaded_shards[shard_idx].meta),
                    mmap: Arc::clone(&loaded_shards[shard_idx].mmap),
                    view,
                },
            );
        }
    }
    tensor_map
}

/// Extract a tensor by name from the unified tensor map.
fn extract_tensor(tensor_map: &TensorMap<'_>, name: &str) -> Result<WeightTensor> {
    let entry = tensor_map
        .get(name)
        .with_context(|| format!("Weight tensor not found: {name}"))?;

    let dtype = convert_dtype(entry.view.dtype()).with_context(|| {
        format!(
            "Unsupported dtype {:?} for tensor {name}",
            entry.view.dtype()
        )
    })?;

    let shape: Vec<usize> = entry.view.shape().to_vec();
    let view_data = entry.view.data();
    let mmap_start = entry.mmap.as_ptr() as usize;
    let mmap_end = mmap_start + entry.mmap.len();
    let data_start = view_data.as_ptr() as usize;
    let data_end = data_start + view_data.len();
    anyhow::ensure!(
        data_start >= mmap_start && data_end <= mmap_end,
        "tensor {name} data does not point into its safetensors mmap"
    );
    let data = WeightData::mmap_slice(
        Arc::clone(&entry.mmap),
        data_start - mmap_start,
        view_data.len(),
    );

    Ok(WeightTensor {
        data,
        shape,
        dtype,
        source: Some(WeightSource {
            shard_path: entry.shard.path.clone(),
            shard_size: entry.shard.size_bytes,
            shard_mtime_ns: entry.shard.modified_ns,
            tensor_name: name.to_owned(),
        }),
    })
}

/// Convert safetensors dtype to our dtype enum.
fn convert_dtype(dt: safetensors::Dtype) -> Result<TensorDType> {
    match dt {
        safetensors::Dtype::F16 => Ok(TensorDType::F16),
        safetensors::Dtype::BF16 => Ok(TensorDType::BF16),
        safetensors::Dtype::F32 => Ok(TensorDType::F32),
        other => bail!("Unsupported dtype: {other:?}"),
    }
}

/// Validate that a tensor has the expected shape.
fn validate_shape(tensor: &WeightTensor, expected: &[usize], name: &str) -> Result<()> {
    if tensor.shape != expected {
        bail!(
            "Shape mismatch for {name}: expected {expected:?}, got {:?}",
            tensor.shape
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GPTQ layer loading
// ---------------------------------------------------------------------------

/// Load one transformer layer's weights from a GPTQ-quantized checkpoint.
///
/// Linear projections (q/k/v/o_proj, gate/up/down_proj, GDN projections) are
/// loaded from packed INT4 (qweight + scales + qzeros) and dequantized to BF16.
/// Non-linear weights (norms, conv1d, a_log, dt_bias) are loaded as-is.
fn load_layer_gptq(
    tensor_map: &TensorMap<'_>,
    prefix: &str,
    layer_idx: usize,
    config: &ModelConfig,
    gptq_config: &GptqConfig,
) -> Result<LayerWeights> {
    let input_layernorm = extract_tensor(tensor_map, &format!("{prefix}input_layernorm.weight"))?;
    validate_shape(
        &input_layernorm,
        &[config.hidden_size],
        &format!("layer {layer_idx} input_layernorm"),
    )?;

    let post_attention_layernorm = extract_tensor(
        tensor_map,
        &format!("{prefix}post_attention_layernorm.weight"),
    )?;
    validate_shape(
        &post_attention_layernorm,
        &[config.hidden_size],
        &format!("layer {layer_idx} post_attn_layernorm"),
    )?;

    let mlp = load_ffn_gptq(tensor_map, prefix, layer_idx, config, gptq_config)?;

    let attention = if config.is_full_attention_layer(layer_idx) {
        AttentionWeights::Full(load_full_attention_gptq(
            tensor_map,
            prefix,
            layer_idx,
            config,
            gptq_config,
        )?)
    } else {
        AttentionWeights::Linear(load_linear_attention_gptq(
            tensor_map,
            prefix,
            layer_idx,
            config,
            gptq_config,
        )?)
    };

    Ok(LayerWeights {
        input_layernorm,
        post_attention_layernorm,
        attention,
        mlp,
    })
}

/// Load and dequantize a single GPTQ-quantized linear projection.
///
/// Looks for `{name}.qweight`, `{name}.scales`, `{name}.qzeros` in the tensor map.
/// Returns a dense BF16 `WeightTensor` with shape `[out_features, in_features]`.
fn load_gptq_linear(
    tensor_map: &TensorMap<'_>,
    name: &str,
    gptq_config: &GptqConfig,
    ctx: &str,
) -> Result<WeightTensor> {
    let qweight = extract_raw_tensor(tensor_map, &format!("{name}.qweight"))
        .with_context(|| format!("{ctx}: qweight"))?;
    let scales = extract_raw_tensor(tensor_map, &format!("{name}.scales"))
        .with_context(|| format!("{ctx}: scales"))?;
    let qzeros = extract_raw_tensor(tensor_map, &format!("{name}.qzeros"))
        .with_context(|| format!("{ctx}: qzeros"))?;

    // Convert scales dtype (GPTQ scales are typically F16)
    let scales_dtype =
        convert_dtype(scales.2).with_context(|| format!("{ctx}: scales dtype {:?}", scales.2))?;

    quantized::dequantize_gptq_weight(
        &qweight.0,
        &qweight.1,
        &scales.0,
        &scales.1,
        scales_dtype,
        &qzeros.0,
        &qzeros.1,
        gptq_config.group_size,
        qweight.3.as_ref().map(|source| WeightSource {
            shard_path: source.shard_path.clone(),
            shard_size: source.shard_size,
            shard_mtime_ns: source.shard_mtime_ns,
            tensor_name: name.to_owned(),
        }),
    )
    .with_context(|| format!("dequantizing {ctx}"))
}

/// Extract raw tensor data (bytes, shape, dtype) without dtype conversion.
/// Used for GPTQ tensors which may be I32 (not supported by our TensorDType).
fn extract_raw_tensor(
    tensor_map: &TensorMap<'_>,
    name: &str,
) -> Result<(
    Vec<u8>,
    Vec<usize>,
    safetensors::Dtype,
    Option<WeightSource>,
)> {
    let entry = tensor_map
        .get(name)
        .with_context(|| format!("Weight tensor not found: {name}"))?;
    Ok((
        entry.view.data().to_vec(),
        entry.view.shape().to_vec(),
        entry.view.dtype(),
        Some(WeightSource {
            shard_path: entry.shard.path.clone(),
            shard_size: entry.shard.size_bytes,
            shard_mtime_ns: entry.shard.modified_ns,
            tensor_name: name.to_owned(),
        }),
    ))
}

/// Load GPTQ-quantized full GQA attention weights.
fn load_full_attention_gptq(
    tensor_map: &TensorMap<'_>,
    prefix: &str,
    layer_idx: usize,
    config: &ModelConfig,
    gptq_config: &GptqConfig,
) -> Result<FullAttentionWeights> {
    let attn = format!("{prefix}self_attn.");
    let ctx = |name: &str| format!("layer {layer_idx} self_attn.{name}");

    let q_proj_dim = config.full_attn_q_proj_dim();
    let q_out_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    let q_proj = load_gptq_linear(
        tensor_map,
        &format!("{attn}q_proj"),
        gptq_config,
        &ctx("q_proj"),
    )?;
    validate_shape(&q_proj, &[q_proj_dim, config.hidden_size], &ctx("q_proj"))?;

    let k_proj = load_gptq_linear(
        tensor_map,
        &format!("{attn}k_proj"),
        gptq_config,
        &ctx("k_proj"),
    )?;
    validate_shape(&k_proj, &[kv_dim, config.hidden_size], &ctx("k_proj"))?;

    let v_proj = load_gptq_linear(
        tensor_map,
        &format!("{attn}v_proj"),
        gptq_config,
        &ctx("v_proj"),
    )?;
    validate_shape(&v_proj, &[kv_dim, config.hidden_size], &ctx("v_proj"))?;

    let o_proj = load_gptq_linear(
        tensor_map,
        &format!("{attn}o_proj"),
        gptq_config,
        &ctx("o_proj"),
    )?;
    validate_shape(&o_proj, &[config.hidden_size, q_out_dim], &ctx("o_proj"))?;

    // QK norms are stored in original precision (1-D, not quantized).
    let q_norm = extract_tensor(tensor_map, &format!("{attn}q_norm.weight"))?;
    validate_shape(&q_norm, &[config.head_dim], &ctx("q_norm"))?;

    let k_norm = extract_tensor(tensor_map, &format!("{attn}k_norm.weight"))?;
    validate_shape(&k_norm, &[config.head_dim], &ctx("k_norm"))?;

    Ok(FullAttentionWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
    })
}

/// Load GPTQ-quantized Gated DeltaNet linear attention weights.
fn load_linear_attention_gptq(
    tensor_map: &TensorMap<'_>,
    prefix: &str,
    layer_idx: usize,
    config: &ModelConfig,
    gptq_config: &GptqConfig,
) -> Result<LinearAttentionWeights> {
    let attn = format!("{prefix}linear_attn.");
    let ctx = |name: &str| format!("layer {layer_idx} linear_attn.{name}");

    let fused_qkv_dim = config.linear_qkv_dim();
    let v_dim = config.linear_v_dim();
    let num_heads = config.linear_num_value_heads;

    // Large projections are GPTQ-quantized.
    let in_proj_qkv = load_gptq_linear(
        tensor_map,
        &format!("{attn}in_proj_qkv"),
        gptq_config,
        &ctx("in_proj_qkv"),
    )?;
    validate_shape(
        &in_proj_qkv,
        &[fused_qkv_dim, config.hidden_size],
        &ctx("in_proj_qkv"),
    )?;

    let in_proj_z = load_gptq_linear(
        tensor_map,
        &format!("{attn}in_proj_z"),
        gptq_config,
        &ctx("in_proj_z"),
    )?;
    validate_shape(&in_proj_z, &[v_dim, config.hidden_size], &ctx("in_proj_z"))?;

    let out_proj = load_gptq_linear(
        tensor_map,
        &format!("{attn}out_proj"),
        gptq_config,
        &ctx("out_proj"),
    )?;
    validate_shape(&out_proj, &[config.hidden_size, v_dim], &ctx("out_proj"))?;

    // Small projections (a, b) — may or may not be quantized in GPTQ models.
    // Try GPTQ first, fall back to dense if qweight not found.
    let in_proj_a = load_gptq_or_dense(
        tensor_map,
        &format!("{attn}in_proj_a"),
        gptq_config,
        &ctx("in_proj_a"),
    )?;
    validate_shape(
        &in_proj_a,
        &[num_heads, config.hidden_size],
        &ctx("in_proj_a"),
    )?;

    let in_proj_b = load_gptq_or_dense(
        tensor_map,
        &format!("{attn}in_proj_b"),
        gptq_config,
        &ctx("in_proj_b"),
    )?;
    validate_shape(
        &in_proj_b,
        &[num_heads, config.hidden_size],
        &ctx("in_proj_b"),
    )?;

    // Non-linear weights: stored in original precision.
    let conv1d = extract_tensor(tensor_map, &format!("{attn}conv1d.weight"))?;
    if conv1d.shape.len() != 3 || conv1d.shape[0] != fused_qkv_dim || conv1d.shape[1] != 1 {
        bail!(
            "Shape mismatch for {}: expected [{fused_qkv_dim}, 1, *], got {:?}",
            ctx("conv1d"),
            conv1d.shape
        );
    }

    let norm = extract_tensor(tensor_map, &format!("{attn}norm.weight"))?;
    validate_shape(&norm, &[config.linear_key_head_dim], &ctx("norm"))?;

    let a_log = extract_tensor(tensor_map, &format!("{attn}A_log"))?;
    if a_log.numel() != num_heads {
        bail!(
            "Element count mismatch for {}: expected {num_heads}, got {} (shape {:?})",
            ctx("A_log"),
            a_log.numel(),
            a_log.shape
        );
    }

    let dt_bias = extract_tensor(tensor_map, &format!("{attn}dt_bias"))?;
    if dt_bias.numel() != num_heads {
        bail!(
            "Element count mismatch for {}: expected {num_heads}, got {} (shape {:?})",
            ctx("dt_bias"),
            dt_bias.numel(),
            dt_bias.shape
        );
    }

    Ok(LinearAttentionWeights {
        in_proj_qkv,
        in_proj_z,
        out_proj,
        in_proj_a,
        in_proj_b,
        conv1d,
        norm,
        a_log,
        dt_bias,
    })
}

/// Load GPTQ-quantized FFN/MLP weights.
fn load_ffn_gptq(
    tensor_map: &TensorMap<'_>,
    prefix: &str,
    layer_idx: usize,
    config: &ModelConfig,
    gptq_config: &GptqConfig,
) -> Result<FfnWeights> {
    let mlp = format!("{prefix}mlp.");
    let ctx = |name: &str| format!("layer {layer_idx} mlp.{name}");

    let gate_proj = load_gptq_linear(
        tensor_map,
        &format!("{mlp}gate_proj"),
        gptq_config,
        &ctx("gate_proj"),
    )?;
    validate_shape(
        &gate_proj,
        &[config.intermediate_size, config.hidden_size],
        &ctx("gate_proj"),
    )?;

    let up_proj = load_gptq_linear(
        tensor_map,
        &format!("{mlp}up_proj"),
        gptq_config,
        &ctx("up_proj"),
    )?;
    validate_shape(
        &up_proj,
        &[config.intermediate_size, config.hidden_size],
        &ctx("up_proj"),
    )?;

    let down_proj = load_gptq_linear(
        tensor_map,
        &format!("{mlp}down_proj"),
        gptq_config,
        &ctx("down_proj"),
    )?;
    validate_shape(
        &down_proj,
        &[config.hidden_size, config.intermediate_size],
        &ctx("down_proj"),
    )?;

    Ok(FfnWeights {
        gate_proj,
        up_proj,
        down_proj,
    })
}

/// Try loading a weight as GPTQ-quantized; fall back to dense if qweight not found.
///
/// Some small projections (e.g., GDN's in_proj_a/b) may not be quantized
/// in certain GPTQ models. This function handles both cases gracefully.
fn load_gptq_or_dense(
    tensor_map: &TensorMap<'_>,
    name: &str,
    gptq_config: &GptqConfig,
    ctx: &str,
) -> Result<WeightTensor> {
    let qweight_name = format!("{name}.qweight");
    if tensor_map.contains_key(qweight_name.as_str()) {
        load_gptq_linear(tensor_map, name, gptq_config, ctx)
    } else {
        // Fall back to dense weight
        extract_tensor(tensor_map, &format!("{name}.weight"))
            .with_context(|| format!("{ctx}: neither GPTQ nor dense weight found"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::Dtype as StDtype;
    use std::collections::HashMap as StdMap;
    use std::io::{Seek, SeekFrom, Write};

    /// Create a minimal safetensors file with the given tensors.
    fn create_test_safetensors(tensors: &[(&str, Vec<usize>, StDtype)]) -> Vec<u8> {
        let mut data_map: StdMap<String, Vec<u8>> = StdMap::new();
        let mut views = Vec::new();

        for (name, shape, dtype) in tensors {
            let numel: usize = shape.iter().product();
            let elem_size = match dtype {
                StDtype::BF16 | StDtype::F16 => 2,
                StDtype::F32 => 4,
                _ => panic!("unsupported dtype in test"),
            };
            let data = vec![0u8; numel * elem_size];
            data_map.insert(name.to_string(), data);
            views.push((name.to_string(), shape.clone(), *dtype));
        }

        let tensor_views: Vec<(String, safetensors::tensor::TensorView<'_>)> = views
            .iter()
            .map(|(name, shape, dtype)| {
                let data = data_map.get(name).unwrap();
                (
                    name.clone(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        let refs: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensor_views
            .iter()
            .map(|(n, v)| (n.as_str(), v.clone()))
            .collect();

        safetensors::tensor::serialize(refs, None).unwrap()
    }

    #[test]
    fn loaded_content_revision_is_path_and_shard_order_independent() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        fs::write(first.path().join("a.safetensors"), b"first shard").unwrap();
        fs::write(first.path().join("b.safetensors"), b"second shard").unwrap();
        fs::write(second.path().join("renamed-2.safetensors"), b"second shard").unwrap();
        fs::write(second.path().join("renamed-1.safetensors"), b"first shard").unwrap();

        let first_mmaps = mmap_shards(&[
            first.path().join("a.safetensors"),
            first.path().join("b.safetensors"),
        ])
        .unwrap();
        let reversed_renamed_mmaps = mmap_shards(&[
            second.path().join("renamed-2.safetensors"),
            second.path().join("renamed-1.safetensors"),
        ])
        .unwrap();

        assert_eq!(
            loaded_shard_content_sha256(&first_mmaps),
            loaded_shard_content_sha256(&reversed_renamed_mmaps)
        );
    }

    #[test]
    fn loaded_content_revision_changes_with_bytes_or_shard_multiplicity() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("a.safetensors");
        let second = dir.path().join("b.safetensors");
        let changed = dir.path().join("changed.safetensors");
        fs::write(&first, b"first shard").unwrap();
        fs::write(&second, b"second shard").unwrap();
        fs::write(&changed, b"second shard changed").unwrap();

        let base = mmap_shards(&[first.clone(), second.clone()]).unwrap();
        let changed_bytes = mmap_shards(&[first.clone(), changed]).unwrap();
        let duplicated = mmap_shards(&[first, second.clone(), second]).unwrap();

        let base_revision = loaded_shard_content_sha256(&base);
        assert_ne!(base_revision, loaded_shard_content_sha256(&changed_bytes));
        assert_ne!(base_revision, loaded_shard_content_sha256(&duplicated));
    }

    /// Build a list of tensor specs for a tiny test model.
    fn tiny_model_tensors(prefix: &str) -> Vec<(String, Vec<usize>, StDtype)> {
        let config = tiny_model_config();
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let vocab = config.vocab_size;

        // Full attention dims
        let q_proj_dim = config.full_attn_q_proj_dim();
        let q_out_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        // Linear attention dims
        let fused_qkv_dim = config.linear_qkv_dim();
        let v_dim = config.linear_v_dim();
        let num_linear_heads = config.linear_num_value_heads;
        let conv_size = config.linear_conv_kernel_dim;

        let mut tensors: Vec<(String, Vec<usize>, StDtype)> = Vec::new();
        let bf16 = StDtype::BF16;

        // Embedding + final norm
        tensors.push((
            format!("{prefix}embed_tokens.weight"),
            vec![vocab, hidden],
            bf16,
        ));
        tensors.push((format!("{prefix}norm.weight"), vec![hidden], bf16));

        // 4 layers: layers 0,1,2 are linear, layer 3 is full attention
        for i in 0..config.num_layers {
            let lp = format!("{prefix}layers.{i}.");
            tensors.push((format!("{lp}input_layernorm.weight"), vec![hidden], bf16));
            tensors.push((
                format!("{lp}post_attention_layernorm.weight"),
                vec![hidden],
                bf16,
            ));
            tensors.push((
                format!("{lp}mlp.gate_proj.weight"),
                vec![intermediate, hidden],
                bf16,
            ));
            tensors.push((
                format!("{lp}mlp.up_proj.weight"),
                vec![intermediate, hidden],
                bf16,
            ));
            tensors.push((
                format!("{lp}mlp.down_proj.weight"),
                vec![hidden, intermediate],
                bf16,
            ));

            if config.is_full_attention_layer(i) {
                // Full attention layer
                tensors.push((
                    format!("{lp}self_attn.q_proj.weight"),
                    vec![q_proj_dim, hidden],
                    bf16,
                ));
                tensors.push((
                    format!("{lp}self_attn.k_proj.weight"),
                    vec![kv_dim, hidden],
                    bf16,
                ));
                tensors.push((
                    format!("{lp}self_attn.v_proj.weight"),
                    vec![kv_dim, hidden],
                    bf16,
                ));
                tensors.push((
                    format!("{lp}self_attn.o_proj.weight"),
                    vec![hidden, q_out_dim],
                    bf16,
                ));
                tensors.push((
                    format!("{lp}self_attn.q_norm.weight"),
                    vec![config.head_dim],
                    bf16,
                ));
                tensors.push((
                    format!("{lp}self_attn.k_norm.weight"),
                    vec![config.head_dim],
                    bf16,
                ));
            } else {
                // Linear attention layer
                tensors.push((
                    format!("{lp}linear_attn.in_proj_qkv.weight"),
                    vec![fused_qkv_dim, hidden],
                    bf16,
                ));
                tensors.push((
                    format!("{lp}linear_attn.in_proj_z.weight"),
                    vec![v_dim, hidden],
                    bf16,
                ));
                tensors.push((
                    format!("{lp}linear_attn.out_proj.weight"),
                    vec![hidden, v_dim],
                    bf16,
                ));
                tensors.push((
                    format!("{lp}linear_attn.in_proj_a.weight"),
                    vec![num_linear_heads, hidden],
                    bf16,
                ));
                tensors.push((
                    format!("{lp}linear_attn.in_proj_b.weight"),
                    vec![num_linear_heads, hidden],
                    bf16,
                ));
                tensors.push((
                    format!("{lp}linear_attn.conv1d.weight"),
                    vec![fused_qkv_dim, 1, conv_size],
                    bf16,
                ));
                tensors.push((
                    format!("{lp}linear_attn.norm.weight"),
                    vec![config.linear_key_head_dim],
                    bf16,
                ));
                tensors.push((
                    format!("{lp}linear_attn.A_log"),
                    vec![num_linear_heads],
                    bf16,
                ));
                tensors.push((
                    format!("{lp}linear_attn.dt_bias"),
                    vec![num_linear_heads],
                    bf16,
                ));
            }
        }

        tensors
    }

    fn tiny_model_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            num_layers: 4,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::BF16,
            num_full_attention_layers: 1,
            full_attention_interval: 4,
            attn_output_gate: true,
            linear_num_key_heads: 4,
            linear_key_head_dim: 8,
            linear_num_value_heads: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 0.25,
        }
    }

    fn load_tiny_source_guard_fixture() -> (tempfile::TempDir, std::path::PathBuf, ModelWeights) {
        let tensors = tiny_model_tensors("model.language_model.");
        let tensor_refs = tensors
            .iter()
            .map(|(name, shape, dtype)| (name.as_str(), shape.clone(), *dtype))
            .collect::<Vec<_>>();
        let bytes = create_test_safetensors(&tensor_refs);
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("model.safetensors");
        fs::write(&source, bytes).unwrap();
        let weights = load_model(dir.path(), &tiny_model_config()).unwrap();
        (dir, source, weights)
    }

    #[test]
    fn source_content_guard_accepts_unchanged_loaded_shards() {
        let (_dir, _source, weights) = load_tiny_source_guard_fixture();

        weights.verify_source_content_unchanged().unwrap();
    }

    #[test]
    fn private_snapshot_is_cleaned_after_last_owner_drops() {
        let (_dir, _source, weights) = load_tiny_source_guard_fixture();
        let snapshot_root = weights
            .source_content_guard
            .as_ref()
            .unwrap()
            .snapshot_root()
            .to_path_buf();
        assert!(snapshot_root.is_dir());
        drop(weights);
        assert!(!snapshot_root.exists());
    }

    #[test]
    fn private_snapshot_explicit_cleanup_is_idempotent_while_owner_is_alive() {
        let (_dir, _source, weights) = load_tiny_source_guard_fixture();
        let cleanup = weights.snapshot_cleanup_handle().unwrap();
        let snapshot_root = cleanup.path().to_path_buf();
        assert!(snapshot_root.is_dir());

        cleanup.cleanup().unwrap();
        assert!(!snapshot_root.exists());
        cleanup.cleanup().unwrap();
        drop(weights);
        assert!(!snapshot_root.exists());
    }

    #[cfg(unix)]
    #[test]
    fn private_snapshot_cleanup_recovers_read_only_directory_after_last_owner() {
        use std::os::unix::fs::PermissionsExt;

        let parent = tempfile::tempdir().unwrap();
        let directory = tempfile::Builder::new()
            .prefix(".kiln-model-snapshot-")
            .tempdir_in(parent.path())
            .unwrap();
        let root = directory.path().to_path_buf();
        let payload = root.join("model.safetensors");
        fs::write(&payload, b"weights").unwrap();
        fs::set_permissions(&payload, fs::Permissions::from_mode(0o400)).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o500)).unwrap();

        let first = Arc::new(ModelSnapshotLease::new(directory));
        let last = Arc::clone(&first);
        drop(first);
        assert!(root.is_dir());
        drop(last);
        assert!(!root.exists());
        assert_eq!(fs::read_dir(parent.path()).unwrap().count(), 0);
    }

    #[test]
    fn shard_discovery_rejects_index_traversal_and_oversize() {
        let traversal = tempfile::tempdir().unwrap();
        fs::write(
            traversal.path().join(INDEX_FILENAME),
            br#"{"weight_map":{"tensor":"../outside.safetensors"}}"#,
        )
        .unwrap();
        let error = discover_shards(traversal.path()).unwrap_err();
        assert!(error.to_string().contains("without traversal"), "{error:#}");

        let oversized = tempfile::tempdir().unwrap();
        let index = File::create(oversized.path().join(INDEX_FILENAME)).unwrap();
        index.set_len(MAX_INDEX_BYTES + 1).unwrap();
        let error = discover_shards(oversized.path()).unwrap_err();
        assert!(error.to_string().contains("safety limit"), "{error:#}");
    }

    #[cfg(unix)]
    #[test]
    fn shard_discovery_rejects_symlink_inputs() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.safetensors");
        fs::write(&target, b"target").unwrap();
        symlink(&target, dir.path().join("model.safetensors")).unwrap();

        let error = discover_shards(dir.path()).unwrap_err();
        assert!(
            error.to_string().contains("must not be a symlink"),
            "{error:#}"
        );
    }

    #[test]
    fn private_snapshot_is_stable_across_same_length_original_mutation() {
        let (_dir, source, weights) = load_tiny_source_guard_fixture();
        let revision = weights.source_content_sha256.clone();
        let mut file = fs::OpenOptions::new().write(true).open(source).unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        file.write_all(&[0xa5]).unwrap();
        file.sync_all().unwrap();

        weights.verify_source_content_unchanged().unwrap();
        assert_eq!(weights.source_content_sha256, revision);
    }

    #[test]
    fn private_snapshot_is_stable_across_original_truncation() {
        let (_dir, source, weights) = load_tiny_source_guard_fixture();
        let file = fs::OpenOptions::new().write(true).open(source).unwrap();
        let original_len = file.metadata().unwrap().len();
        file.set_len(original_len - 1).unwrap();

        weights.verify_source_content_unchanged().unwrap();
    }

    #[test]
    fn source_content_guard_verifies_shards_without_extracted_model_tensors() {
        let model_tensors = tiny_model_tensors("model.language_model.");
        let model_refs = model_tensors
            .iter()
            .map(|(name, shape, dtype)| (name.as_str(), shape.clone(), *dtype))
            .collect::<Vec<_>>();
        let ignored_refs = [("model.visual.unused", vec![1], StDtype::BF16)];
        let dir = tempfile::tempdir().unwrap();
        let model_shard = dir.path().join("model-00001-of-00002.safetensors");
        let ignored_shard = dir.path().join("model-00002-of-00002.safetensors");
        fs::write(&model_shard, create_test_safetensors(&model_refs)).unwrap();
        fs::write(&ignored_shard, create_test_safetensors(&ignored_refs)).unwrap();

        let mut weight_map = serde_json::Map::new();
        for (name, _, _) in &model_tensors {
            weight_map.insert(
                name.clone(),
                serde_json::Value::String("model-00001-of-00002.safetensors".to_string()),
            );
        }
        weight_map.insert(
            "model.visual.unused".to_string(),
            serde_json::Value::String("model-00002-of-00002.safetensors".to_string()),
        );
        fs::write(
            dir.path().join(INDEX_FILENAME),
            serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
        )
        .unwrap();

        let weights = load_model(dir.path(), &tiny_model_config()).unwrap();
        weights.verify_source_content_unchanged().unwrap();

        let mut file = fs::OpenOptions::new()
            .write(true)
            .open(ignored_shard)
            .unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        file.write_all(&[0xa5]).unwrap();
        file.sync_all().unwrap();

        // Original replacement is irrelevant after the private copy.
        weights.verify_source_content_unchanged().unwrap();

        // The post-upload guard still covers every copied shard, including
        // shards from which no model tensor was extracted.
        let snapshot_root = weights
            .source_content_guard
            .as_ref()
            .unwrap()
            .snapshot_root()
            .to_path_buf();
        let snapshot_ignored = snapshot_root.join("model-00002-of-00002.safetensors");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&snapshot_ignored, fs::Permissions::from_mode(0o600)).unwrap();
        }
        #[cfg(not(unix))]
        {
            let mut permissions = fs::metadata(&snapshot_ignored).unwrap().permissions();
            permissions.set_readonly(false);
            fs::set_permissions(&snapshot_ignored, permissions).unwrap();
        }
        let mut snapshot_file = fs::OpenOptions::new()
            .write(true)
            .open(snapshot_ignored)
            .unwrap();
        snapshot_file.seek(SeekFrom::End(-1)).unwrap();
        snapshot_file.write_all(&[0xa5]).unwrap();
        snapshot_file.sync_all().unwrap();
        assert!(weights.verify_source_content_unchanged().is_err());
    }

    #[test]
    fn test_load_single_shard() {
        let tensors = tiny_model_tensors("model.language_model.");
        let tensor_refs: Vec<(&str, Vec<usize>, StDtype)> = tensors
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();

        let bytes = create_test_safetensors(&tensor_refs);

        let dir = tempfile::tempdir().unwrap();
        let safetensors_path = dir.path().join("model.safetensors");
        fs::write(&safetensors_path, &bytes).unwrap();

        let config = tiny_model_config();
        let weights = load_model(dir.path(), &config).unwrap();

        // Check structure.
        assert_eq!(weights.layers.len(), 4);

        // Layers 0, 1, 2 should be linear attention.
        for i in 0..3 {
            assert!(
                matches!(weights.layers[i].attention, AttentionWeights::Linear(_)),
                "Layer {i} should be linear attention"
            );
        }

        // Layer 3 should be full attention.
        assert!(
            matches!(weights.layers[3].attention, AttentionWeights::Full(_)),
            "Layer 3 should be full attention"
        );

        // Check embedding shape.
        assert_eq!(weights.embedding.embed_tokens.shape, vec![256, 64]);
        assert_eq!(weights.embedding.embed_tokens.dtype, TensorDType::BF16);

        // Check final norm.
        assert_eq!(weights.final_norm.shape, vec![64]);

        // Check total params > 0.
        assert!(weights.total_params() > 0);
        assert!(weights.total_bytes() > 0);
    }

    #[test]
    fn test_load_bare_prefix() {
        // Test with "model." prefix (text-only checkpoint).
        let tensors = tiny_model_tensors("model.");
        let tensor_refs: Vec<(&str, Vec<usize>, StDtype)> = tensors
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();

        let bytes = create_test_safetensors(&tensor_refs);

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();

        let config = tiny_model_config();
        let weights = load_model(dir.path(), &config).unwrap();
        assert_eq!(weights.layers.len(), 4);
    }

    #[test]
    fn test_load_sharded() {
        let all_tensors = tiny_model_tensors("model.language_model.");
        let mid = all_tensors.len() / 2;

        let shard1_tensors: Vec<(&str, Vec<usize>, StDtype)> = all_tensors[..mid]
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();
        let shard2_tensors: Vec<(&str, Vec<usize>, StDtype)> = all_tensors[mid..]
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();

        let bytes1 = create_test_safetensors(&shard1_tensors);
        let bytes2 = create_test_safetensors(&shard2_tensors);

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model-00001-of-00002.safetensors"), &bytes1).unwrap();
        fs::write(dir.path().join("model-00002-of-00002.safetensors"), &bytes2).unwrap();

        // Create index file.
        let mut weight_map = serde_json::Map::new();
        for (name, _, _) in &all_tensors[..mid] {
            weight_map.insert(
                name.clone(),
                serde_json::Value::String("model-00001-of-00002.safetensors".into()),
            );
        }
        for (name, _, _) in &all_tensors[mid..] {
            weight_map.insert(
                name.clone(),
                serde_json::Value::String("model-00002-of-00002.safetensors".into()),
            );
        }
        let index = serde_json::json!({ "weight_map": weight_map });
        fs::write(
            dir.path().join(INDEX_FILENAME),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let config = tiny_model_config();
        let weights = load_model(dir.path(), &config).unwrap();
        assert_eq!(weights.layers.len(), 4);
    }

    #[test]
    fn test_shape_mismatch_detected() {
        let mut tensors = tiny_model_tensors("model.language_model.");
        // Corrupt embedding shape.
        tensors[0].1 = vec![999, 64];

        let tensor_refs: Vec<(&str, Vec<usize>, StDtype)> = tensors
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();

        let bytes = create_test_safetensors(&tensor_refs);

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();

        let config = tiny_model_config();
        let result = load_model(dir.path(), &config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Shape mismatch"),
            "Error should mention shape mismatch: {err}"
        );
    }

    #[test]
    fn test_missing_tensor_detected() {
        // Create tensors but skip the embedding.
        let tensors = tiny_model_tensors("model.language_model.");
        let tensor_refs: Vec<(&str, Vec<usize>, StDtype)> = tensors
            .iter()
            .skip(1) // skip embed_tokens
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();

        let bytes = create_test_safetensors(&tensor_refs);

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();

        let config = tiny_model_config();
        let result = load_model(dir.path(), &config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found"),
            "Error should mention missing tensor: {err}"
        );
    }

    /// Append MTP tensor specs for the tiny test model.
    ///
    /// `mtp_prefix` controls the prefix used for the MTP weights (`"mtp."`
    /// for Qwen3.5-4B stock, `"{base}mtp."` for VL-wrapped checkpoints).
    /// `final_ln_name` is the leaf used for the head RMSNorm — either
    /// `"norm"` (Qwen3.5-4B stock) or `"final_layernorm"` (older convention).
    fn tiny_mtp_tensors(
        mtp_prefix: &str,
        final_ln_name: &str,
    ) -> Vec<(String, Vec<usize>, StDtype)> {
        let config = tiny_model_config();
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let q_proj_dim = config.full_attn_q_proj_dim();
        let q_out_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let bf16 = StDtype::BF16;

        let mut tensors: Vec<(String, Vec<usize>, StDtype)> = Vec::new();
        tensors.push((
            format!("{mtp_prefix}fc.weight"),
            vec![hidden, 2 * hidden],
            bf16,
        ));
        tensors.push((
            format!("{mtp_prefix}pre_fc_norm_embedding.weight"),
            vec![hidden],
            bf16,
        ));
        tensors.push((
            format!("{mtp_prefix}pre_fc_norm_hidden.weight"),
            vec![hidden],
            bf16,
        ));
        tensors.push((
            format!("{mtp_prefix}{final_ln_name}.weight"),
            vec![hidden],
            bf16,
        ));

        // One full-attention transformer layer at mtp.layers.0.*
        let lp = format!("{mtp_prefix}layers.0.");
        tensors.push((format!("{lp}input_layernorm.weight"), vec![hidden], bf16));
        tensors.push((
            format!("{lp}post_attention_layernorm.weight"),
            vec![hidden],
            bf16,
        ));
        tensors.push((
            format!("{lp}mlp.gate_proj.weight"),
            vec![intermediate, hidden],
            bf16,
        ));
        tensors.push((
            format!("{lp}mlp.up_proj.weight"),
            vec![intermediate, hidden],
            bf16,
        ));
        tensors.push((
            format!("{lp}mlp.down_proj.weight"),
            vec![hidden, intermediate],
            bf16,
        ));
        tensors.push((
            format!("{lp}self_attn.q_proj.weight"),
            vec![q_proj_dim, hidden],
            bf16,
        ));
        tensors.push((
            format!("{lp}self_attn.k_proj.weight"),
            vec![kv_dim, hidden],
            bf16,
        ));
        tensors.push((
            format!("{lp}self_attn.v_proj.weight"),
            vec![kv_dim, hidden],
            bf16,
        ));
        tensors.push((
            format!("{lp}self_attn.o_proj.weight"),
            vec![hidden, q_out_dim],
            bf16,
        ));
        tensors.push((
            format!("{lp}self_attn.q_norm.weight"),
            vec![config.head_dim],
            bf16,
        ));
        tensors.push((
            format!("{lp}self_attn.k_norm.weight"),
            vec![config.head_dim],
            bf16,
        ));
        tensors
    }

    /// Load a tiny model with the given base prefix + MTP tensors under
    /// `mtp_prefix` with `final_ln_name` naming, returning the loaded weights.
    fn load_tiny_with_mtp(
        base_prefix: &str,
        mtp_prefix: &str,
        final_ln_name: &str,
    ) -> ModelWeights {
        let mut tensors = tiny_model_tensors(base_prefix);
        tensors.extend(tiny_mtp_tensors(mtp_prefix, final_ln_name));

        let tensor_refs: Vec<(&str, Vec<usize>, StDtype)> = tensors
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();
        let bytes = create_test_safetensors(&tensor_refs);

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();

        let config = tiny_model_config();
        load_model(dir.path(), &config).unwrap()
    }

    #[test]
    fn test_load_mtp_vl_prefix_with_norm() {
        // VL-wrapped layout: mtp tensors nested under the LM prefix,
        // final RMSNorm is `norm.weight`.
        let weights =
            load_tiny_with_mtp("model.language_model.", "model.language_model.mtp.", "norm");
        let mtp = weights.mtp.expect("MTP weights should have loaded");
        assert_eq!(mtp.fc.shape, vec![64, 128]);
        assert_eq!(mtp.final_layernorm.shape, vec![64]);
        assert!(matches!(mtp.layer.attention, AttentionWeights::Full(_)));
    }

    #[test]
    fn test_load_mtp_bare_prefix_qwen35() {
        // Qwen3.5-4B stock layout: LM under `model.language_model.` prefix
        // but MTP tensors at bare `mtp.*` with `norm.weight` head norm.
        let weights = load_tiny_with_mtp("model.language_model.", "mtp.", "norm");
        let mtp = weights
            .mtp
            .expect("MTP weights should have loaded (bare prefix)");
        assert_eq!(mtp.fc.shape, vec![64, 128]);
        assert_eq!(mtp.final_layernorm.shape, vec![64]);
        assert!(matches!(mtp.layer.attention, AttentionWeights::Full(_)));
    }

    #[test]
    fn test_load_mtp_final_layernorm_alias() {
        // Older convention / vLLM naming: final RMSNorm is
        // `final_layernorm.weight`. Must still load.
        let weights = load_tiny_with_mtp("model.language_model.", "mtp.", "final_layernorm");
        let mtp = weights
            .mtp
            .expect("MTP weights should have loaded with final_layernorm alias");
        assert_eq!(mtp.final_layernorm.shape, vec![64]);
    }

    #[test]
    fn test_gpu_weights_defer_mtp_upload_until_first_use() {
        let config = tiny_model_config();
        let weights = load_tiny_with_mtp("model.language_model.", "mtp.", "norm");
        // #1082: GpuWeights::from_model_weights takes a kt `Device`.
        let device = kiln_tensor::Device::Cpu;
        let gpu_weights =
            crate::forward::GpuWeights::from_model_weights(&weights, &config, &device).unwrap();

        let slot = gpu_weights
            .mtp
            .as_ref()
            .expect("GPU weights should expose MTP support");
        assert!(
            !slot.is_uploaded(),
            "MTP tensors should not upload during base model load"
        );

        let mtp = gpu_weights.mtp_weights().unwrap();
        // #1082: kt has no `dims1`; assert the full rank-1 shape (stronger).
        assert_eq!(mtp.final_layernorm.dims(), &[64]);
        assert!(
            gpu_weights.mtp.as_ref().unwrap().is_uploaded(),
            "first native-MTP access should materialize the GPU tensors"
        );
    }

    #[test]
    fn test_load_mtp_can_be_deferred_to_first_use() {
        let mut tensors = tiny_model_tensors("model.language_model.");
        tensors.extend(tiny_mtp_tensors("mtp.", "norm"));

        let tensor_refs: Vec<(&str, Vec<usize>, StDtype)> = tensors
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();
        let bytes = create_test_safetensors(&tensor_refs);

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();

        let config = tiny_model_config();
        let weights =
            load_model_with_options(dir.path(), &config, LoadModelOptions { load_mtp: false })
                .unwrap();
        assert!(weights.mtp.is_none(), "MTP should stay off the eager path");
        assert!(
            weights.deferred_mtp.is_some(),
            "MTP presence should still be visible via deferred source"
        );

        // #1082: GpuWeights::from_model_weights takes a kt `Device`.
        let device = kiln_tensor::Device::Cpu;
        let gpu_weights =
            crate::forward::GpuWeights::from_model_weights(&weights, &config, &device).unwrap();
        let slot = gpu_weights
            .mtp
            .as_ref()
            .expect("deferred source should still expose native MTP support");
        assert!(!slot.is_uploaded(), "deferred MTP must remain lazy");

        let mtp = gpu_weights.mtp_weights().unwrap();
        // #1082: kt has no `dims1`; assert the full rank-1 shape (stronger).
        assert_eq!(mtp.final_layernorm.dims(), &[64]);
        assert!(
            gpu_weights.mtp.as_ref().unwrap().is_uploaded(),
            "first native-MTP access should load deferred CPU+GPU tensors"
        );
    }

    #[test]
    fn test_load_without_mtp_leaves_none() {
        // Base checkpoint without MTP should leave mtp = None and load fine.
        let tensors = tiny_model_tensors("model.language_model.");
        let tensor_refs: Vec<(&str, Vec<usize>, StDtype)> = tensors
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();
        let bytes = create_test_safetensors(&tensor_refs);

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();

        let config = tiny_model_config();
        let weights = load_model(dir.path(), &config).unwrap();
        assert!(
            weights.mtp.is_none(),
            "MTP should be None when tensors are absent"
        );
        assert!(
            weights.deferred_mtp.is_none(),
            "deferred MTP should also be absent when tensors are absent"
        );
    }

    // -----------------------------------------------------------------------
    // GPTQ loading tests
    // -----------------------------------------------------------------------

    /// Create a safetensors file with custom binary data for each tensor.
    /// Unlike `create_test_safetensors` which fills with zeros, this takes
    /// pre-built byte arrays.
    fn create_safetensors_with_data(tensors: &[(&str, Vec<usize>, StDtype, Vec<u8>)]) -> Vec<u8> {
        let tensor_views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(name, shape, dtype, data)| {
                (
                    name.to_string(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        let refs: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensor_views
            .iter()
            .map(|(n, v)| (n.as_str(), v.clone()))
            .collect();

        safetensors::tensor::serialize(refs, None).unwrap()
    }

    /// Pack `values` (each 0..15) into u32 words, 8 values per word.
    fn pack_int4(values: &[u8]) -> Vec<u8> {
        let mut words: Vec<u32> = Vec::new();
        for chunk in values.chunks(8) {
            let mut packed = 0u32;
            for (i, &v) in chunk.iter().enumerate() {
                packed |= (v as u32 & 0xF) << (i * 4);
            }
            words.push(packed);
        }
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Create GPTQ tensor specs for a single linear layer.
    /// Returns: (name_prefix, qweight_data, scales_data, qzeros_data, tensors_spec)
    fn gptq_linear_tensors(
        name: &str,
        out_features: usize,
        in_features: usize,
        group_size: usize,
    ) -> Vec<(&'static str, String, Vec<usize>, StDtype, Vec<u8>)> {
        let pack_factor = 8;
        let in_packed = in_features / pack_factor;
        let num_groups = in_features / group_size;
        let out_packed = out_features / pack_factor;

        // qweight: all 5s (INT4 value = 5)
        let total_int4_values = in_features * out_features;
        let qweight_values = vec![5u8; total_int4_values];
        // Pack column-major: qweight[in/8, out]
        let mut qweight_packed = Vec::new();
        for pack_row in 0..in_packed {
            for out_idx in 0..out_features {
                let mut packed = 0u32;
                for bit in 0..pack_factor {
                    let in_idx = pack_row * pack_factor + bit;
                    let val = qweight_values[in_idx * out_features + out_idx] as u32;
                    packed |= (val & 0xF) << (bit * 4);
                }
                qweight_packed.push(packed);
            }
        }
        let qweight_bytes: Vec<u8> = qweight_packed
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();

        // qzeros: all 8s (midpoint)
        let mut qzeros_packed = Vec::new();
        for _group in 0..num_groups {
            for _pack_col in 0..out_packed {
                let mut packed = 0u32;
                for bit in 0..pack_factor {
                    packed |= 8u32 << (bit * 4);
                }
                qzeros_packed.push(packed);
            }
        }
        let qzeros_bytes: Vec<u8> = qzeros_packed.iter().flat_map(|w| w.to_le_bytes()).collect();

        // scales: all 1.0 in F16
        // F16 1.0 = 0x3C00
        let scales_bytes: Vec<u8> =
            std::iter::repeat_n(0x3C00u16.to_le_bytes(), num_groups * out_features)
                .flatten()
                .collect();

        // We leak the name string to get 'static lifetime for the test
        let qweight_name = format!("{name}.qweight");
        let scales_name = format!("{name}.scales");
        let qzeros_name = format!("{name}.qzeros");

        vec![
            (
                "qweight",
                qweight_name,
                vec![in_packed, out_features],
                StDtype::I32,
                qweight_bytes,
            ),
            (
                "scales",
                scales_name,
                vec![num_groups, out_features],
                StDtype::F16,
                scales_bytes,
            ),
            (
                "qzeros",
                qzeros_name,
                vec![num_groups, out_packed],
                StDtype::I32,
                qzeros_bytes,
            ),
        ]
    }

    /// Build GPTQ tensor specs for the tiny test model.
    fn tiny_gptq_model_tensors(prefix: &str) -> Vec<(String, Vec<usize>, StDtype, Vec<u8>)> {
        let config = tiny_model_config();
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let vocab = config.vocab_size;
        let group_size = 8; // Small group size for tiny model

        let bf16 = StDtype::BF16;

        let mut tensors: Vec<(String, Vec<usize>, StDtype, Vec<u8>)> = Vec::new();

        // Helper functions (free functions to avoid closure borrow conflicts).
        fn make_bf16_tensor(
            name: String,
            shape: Vec<usize>,
            bf16: StDtype,
        ) -> (String, Vec<usize>, StDtype, Vec<u8>) {
            let numel: usize = shape.iter().product();
            (name, shape, bf16, vec![0u8; numel * 2])
        }

        fn make_gptq_tensors(
            name: &str,
            out: usize,
            inp: usize,
            group_size: usize,
        ) -> Vec<(String, Vec<usize>, StDtype, Vec<u8>)> {
            let pack_factor = 8usize;
            let in_packed = inp / pack_factor;
            let num_groups = inp / group_size;

            let mut result = Vec::new();

            // qweight: [in/8, out], all 5s
            let mut qweight_words = Vec::new();
            for _row in 0..in_packed {
                for _col in 0..out {
                    let packed: u32 = (0..pack_factor as u32).map(|i| 5u32 << (i * 4)).sum();
                    qweight_words.push(packed);
                }
            }
            let qweight_bytes: Vec<u8> =
                qweight_words.iter().flat_map(|w| w.to_le_bytes()).collect();
            result.push((
                format!("{name}.qweight"),
                vec![in_packed, out],
                StDtype::I32,
                qweight_bytes,
            ));

            // scales: [num_groups, out], all F16 1.0
            let scales_bytes: Vec<u8> = vec![0x00, 0x3C].repeat(num_groups * out);
            result.push((
                format!("{name}.scales"),
                vec![num_groups, out],
                StDtype::F16,
                scales_bytes,
            ));

            // qzeros: [num_groups, out/8], all 8s
            let out_packed_actual = (out + pack_factor - 1) / pack_factor;
            let mut qzeros_words = Vec::new();
            for _ in 0..num_groups * out_packed_actual {
                let packed: u32 = (0..pack_factor as u32).map(|i| 8u32 << (i * 4)).sum();
                qzeros_words.push(packed);
            }
            let qzeros_bytes: Vec<u8> = qzeros_words.iter().flat_map(|w| w.to_le_bytes()).collect();
            result.push((
                format!("{name}.qzeros"),
                vec![num_groups, out_packed_actual],
                StDtype::I32,
                qzeros_bytes,
            ));

            result
        }

        macro_rules! add_bf16 {
            ($name:expr, $shape:expr) => {
                tensors.push(make_bf16_tensor($name, $shape, bf16));
            };
        }
        macro_rules! add_gptq_linear {
            ($name:expr, $out:expr, $inp:expr) => {
                tensors.extend(make_gptq_tensors($name, $out, $inp, group_size));
            };
        }

        // Full attention dims
        let q_proj_dim = config.full_attn_q_proj_dim();
        let q_out_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        // Linear attention dims
        let fused_qkv_dim = config.linear_qkv_dim();
        let v_dim = config.linear_v_dim();
        let num_linear_heads = config.linear_num_value_heads;
        let conv_size = config.linear_conv_kernel_dim;

        // Embedding + final norm (not quantized)
        add_bf16!(format!("{prefix}embed_tokens.weight"), vec![vocab, hidden]);
        add_bf16!(format!("{prefix}norm.weight"), vec![hidden]);

        for i in 0..config.num_layers {
            let lp = format!("{prefix}layers.{i}.");

            // Norms (not quantized)
            add_bf16!(format!("{lp}input_layernorm.weight"), vec![hidden]);
            add_bf16!(format!("{lp}post_attention_layernorm.weight"), vec![hidden]);

            // MLP (quantized)
            add_gptq_linear!(&format!("{lp}mlp.gate_proj"), intermediate, hidden);
            add_gptq_linear!(&format!("{lp}mlp.up_proj"), intermediate, hidden);
            add_gptq_linear!(&format!("{lp}mlp.down_proj"), hidden, intermediate);

            if config.is_full_attention_layer(i) {
                // Full attention projections (quantized)
                add_gptq_linear!(&format!("{lp}self_attn.q_proj"), q_proj_dim, hidden);
                add_gptq_linear!(&format!("{lp}self_attn.k_proj"), kv_dim, hidden);
                add_gptq_linear!(&format!("{lp}self_attn.v_proj"), kv_dim, hidden);
                add_gptq_linear!(&format!("{lp}self_attn.o_proj"), hidden, q_out_dim);
                // QK norms (not quantized)
                add_bf16!(
                    format!("{lp}self_attn.q_norm.weight"),
                    vec![config.head_dim]
                );
                add_bf16!(
                    format!("{lp}self_attn.k_norm.weight"),
                    vec![config.head_dim]
                );
            } else {
                // Linear attention: large projections quantized
                add_gptq_linear!(
                    &format!("{lp}linear_attn.in_proj_qkv"),
                    fused_qkv_dim,
                    hidden
                );
                add_gptq_linear!(&format!("{lp}linear_attn.in_proj_z"), v_dim, hidden);
                add_gptq_linear!(&format!("{lp}linear_attn.out_proj"), hidden, v_dim);
                // Small projections as dense (a, b have small out dim)
                add_bf16!(
                    format!("{lp}linear_attn.in_proj_a.weight"),
                    vec![num_linear_heads, hidden]
                );
                add_bf16!(
                    format!("{lp}linear_attn.in_proj_b.weight"),
                    vec![num_linear_heads, hidden]
                );
                // Non-linear weights (not quantized)
                add_bf16!(
                    format!("{lp}linear_attn.conv1d.weight"),
                    vec![fused_qkv_dim, 1, conv_size]
                );
                add_bf16!(
                    format!("{lp}linear_attn.norm.weight"),
                    vec![config.linear_key_head_dim]
                );
                add_bf16!(format!("{lp}linear_attn.A_log"), vec![num_linear_heads]);
                add_bf16!(format!("{lp}linear_attn.dt_bias"), vec![num_linear_heads]);
            }
        }

        tensors
    }

    #[test]
    fn test_load_gptq_model() {
        let tensors = tiny_gptq_model_tensors("model.language_model.");

        // Build safetensors with custom data
        let tensor_views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(name, shape, dtype, data)| {
                (
                    name.clone(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        let refs: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensor_views
            .iter()
            .map(|(n, v)| (n.as_str(), v.clone()))
            .collect();
        let bytes = safetensors::tensor::serialize(refs, None).unwrap();

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();

        // Write quantize_config.json
        fs::write(
            dir.path().join("quantize_config.json"),
            r#"{"bits": 4, "group_size": 8, "sym": false, "desc_act": false}"#,
        )
        .unwrap();

        let config = tiny_model_config();
        let weights = load_model(dir.path(), &config).unwrap();

        // Basic structural checks
        assert_eq!(weights.layers.len(), 4);
        assert_eq!(weights.embedding.embed_tokens.shape, vec![256, 64]);
        assert_eq!(weights.final_norm.shape, vec![64]);

        // Layer types
        for i in 0..3 {
            assert!(
                matches!(weights.layers[i].attention, AttentionWeights::Linear(_)),
                "Layer {i} should be linear attention"
            );
        }
        assert!(
            matches!(weights.layers[3].attention, AttentionWeights::Full(_)),
            "Layer 3 should be full attention"
        );

        // Check that dequantized weights have correct shapes.
        // MLP gate_proj: [intermediate, hidden] = [128, 64]
        assert_eq!(weights.layers[0].mlp.gate_proj.shape, vec![128, 64]);
        assert_eq!(weights.layers[0].mlp.gate_proj.dtype, TensorDType::BF16);

        // Full attention q_proj: [q_proj_dim, hidden]
        if let AttentionWeights::Full(ref attn) = weights.layers[3].attention {
            // q_proj_dim = 4*16*2 = 128 (with gate), hidden = 64
            assert_eq!(attn.q_proj.shape, vec![128, 64]);
            assert_eq!(attn.q_proj.dtype, TensorDType::BF16);
        }

        // Linear attention in_proj_qkv
        if let AttentionWeights::Linear(ref attn) = weights.layers[0].attention {
            let qkv_dim = config.linear_qkv_dim();
            assert_eq!(attn.in_proj_qkv.shape, vec![qkv_dim, 64]);
            assert_eq!(attn.in_proj_qkv.dtype, TensorDType::BF16);
        }

        // Verify dequantized values are reasonable.
        // With weight=5, zero=8, scale=1.0: expected = (5-8)*1.0 = -3.0
        let gate_data = weights.layers[0].mlp.gate_proj.as_bytes();
        let first_bf16 = u16::from_le_bytes([gate_data[0], gate_data[1]]);
        let first_val = f32::from_bits((first_bf16 as u32) << 16);
        assert!(
            (first_val - (-3.0)).abs() < 0.1,
            "Expected dequantized value ~-3.0, got {first_val}"
        );

        assert!(weights.total_params() > 0);
        assert!(weights.total_bytes() > 0);
    }

    #[test]
    fn test_gptq_config_autodetect() {
        // Without quantize_config.json, should load as dense
        let tensors = tiny_model_tensors("model.language_model.");
        let tensor_refs: Vec<(&str, Vec<usize>, StDtype)> = tensors
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), *d))
            .collect();
        let bytes = create_test_safetensors(&tensor_refs);

        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();
        // No quantize_config.json

        let config = tiny_model_config();
        let weights = load_model(dir.path(), &config).unwrap();
        assert_eq!(weights.layers.len(), 4);
    }
}
