//! Immutable identity for Kiln's base-model prompt-logprob service.
//!
//! The model alias returned by OpenAI-compatible endpoints is not a content
//! identity. This module binds the exact loader-owned model bytes, numeric
//! tokenizer vocabulary, tokenizer processing config, model config, backend,
//! startup-resolved streaming-prefill policy, and prompt-logprob semantics into
//! the canonical identity shared with remote training clients.

use anyhow::{Context, Result, ensure};
use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::lora_loader::{LoraSourceIdentity, LoraWeights};
use kiln_model::{StreamingPrefillExecutionPolicy, StreamingPrefillMode};
use kiln_tensor::Device;
use kiln_train::{TeacherAdapterIdentityV1, TeacherIdentityV1};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::{self, Read};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const RUNTIME_COMMAND_TIMEOUT: Duration = Duration::from_secs(5);
const RUNTIME_COMMAND_CAPTURE_BYTES: usize = 1024 * 1024;
const RUNTIME_FILE_CAPTURE_BYTES: usize = 2 * 1024 * 1024;

pub(crate) const MAX_COMPLETION_PROMPT_LOGPROBS: usize = 256;
pub(crate) const MAX_COMPLETION_PROMPT_LOGPROB_CANDIDATES: usize = 65_536;
pub(crate) const PROMPT_LOGPROB_PROJECTION_BYTE_BUDGET: usize = 64 * 1024 * 1024;
pub(crate) const MAX_PROMPT_LOGPROB_PROJECTION_CHUNK_TOKENS: usize = 32;
pub(crate) const MAX_COMPLETION_PROMPT_TOKENS: usize = 4096;

const INFERENCE_CONTRACT_SCHEMA: &str = "kiln.prompt-logprobs.inference-config.v2";
const CAUSAL_ALIGNMENT: &str =
    "response prompt_logprobs[p] scores observed prompt token p from logits row p-1";
const LOGPROB_SEMANTICS: &str = "raw model logits; full-vocabulary log-softmax; finite f32 wire values; rank from original logits";
const CANDIDATE_SEMANTICS: &str =
    "observed token plus requested top-k; k entries when observed is top-k, otherwise k+1";

#[derive(Serialize)]
struct PromptLogprobHardCaps {
    max_top_k: usize,
    max_prompt_tokens: usize,
    max_response_candidates: usize,
    projection_byte_budget: usize,
    max_projection_chunk_tokens: usize,
}

#[derive(Serialize)]
struct KilnInferenceContract<'a> {
    schema: &'static str,
    model_config: &'a ModelConfig,
    backend: &'a str,
    streaming_prefill: StreamingPrefillInferenceContract,
    executable_sha256: &'a str,
    numerical_runtime_sha256: &'a str,
    causal_alignment: &'static str,
    logprob_semantics: &'static str,
    candidate_semantics: &'static str,
    hard_caps: PromptLogprobHardCaps,
}

#[derive(Serialize)]
struct StreamingPrefillInferenceContract {
    mode: &'static str,
    threshold_tokens: Option<usize>,
    base_tile_tokens: usize,
    tape_tile_tokens: usize,
    detached_full_attn_tile_tokens: usize,
    detached_full_attn_boundary_tile_tokens: usize,
    detached_full_attn_tape_replay_tile_tokens: usize,
    last_token_lm_head: bool,
}

impl From<StreamingPrefillExecutionPolicy> for StreamingPrefillInferenceContract {
    fn from(policy: StreamingPrefillExecutionPolicy) -> Self {
        let mode = match policy.mode() {
            StreamingPrefillMode::Auto => "auto",
            StreamingPrefillMode::Enabled => "enabled",
            StreamingPrefillMode::Disabled => "disabled",
        };
        Self {
            mode,
            threshold_tokens: policy.threshold_tokens(),
            base_tile_tokens: policy.base_tile_tokens(),
            tape_tile_tokens: policy.tape_tile_tokens(),
            detached_full_attn_tile_tokens: policy.detached_full_attn_tile_tokens(),
            detached_full_attn_boundary_tile_tokens: policy
                .detached_full_attn_boundary_tile_tokens(),
            detached_full_attn_tape_replay_tile_tokens: policy
                .detached_full_attn_tape_replay_tile_tokens(),
            last_token_lm_head: policy.last_token_lm_head(),
        }
    }
}

/// Build the canonical base-model identity published by Kiln completions.
///
/// `base_model_source_sha256` must be the loader-owned digest captured before
/// parsing and verified again immediately after GPU upload. `streaming_prefill`
/// must be the exact immutable policy installed on the scoring runner. Prefixed
/// hashes are normalized only after strict lowercase SHA-256 validation.
// The flat argument list mirrors the CLI-flag/API field set 1:1; a parameter struct would obscure that correspondence, and changing the signature would be a breaking API change.
#[allow(clippy::too_many_arguments)]
pub fn build_base_teacher_identity(
    served_model_id: &str,
    base_model_source_sha256: &str,
    tokenizer: &KilnTokenizer,
    model_config: &ModelConfig,
    backend: &str,
    streaming_prefill: StreamingPrefillExecutionPolicy,
    executable_sha256: &str,
    numerical_runtime_sha256: &str,
) -> Result<TeacherIdentityV1> {
    ensure!(
        !backend.is_empty() && backend.trim() == backend && !backend.chars().any(char::is_control),
        "inference backend name must be non-empty, trimmed, and contain no control characters"
    );
    let tokenizer_vocab_size = tokenizer.vocab_size();
    ensure!(
        tokenizer_vocab_size <= model_config.vocab_size,
        "tokenizer vocabulary entry count {tokenizer_vocab_size} exceeds model vocabulary size {}",
        model_config.vocab_size
    );
    let max_token_id = tokenizer
        .max_token_id()
        .context("tokenizer vocabulary must not be empty")?;
    ensure!(
        usize::try_from(max_token_id)
            .map(|token_id| token_id < model_config.vocab_size)
            .unwrap_or(false),
        "tokenizer maximum token ID {max_token_id} is outside model vocabulary size {}",
        model_config.vocab_size
    );

    let base_model_sha256 = strip_sha256_prefix(
        "loader-owned base model content revision",
        base_model_source_sha256,
    )?;
    let tokenizer_vocab_sha256 = strip_sha256_prefix(
        "tokenizer vocabulary identity",
        &tokenizer.vocab_identity_sha256(),
    )?;
    let tokenizer_config_sha256 = strip_sha256_prefix(
        "tokenizer config identity",
        &tokenizer
            .tokenizer_config_sha256()
            .context("failed to serialize tokenizer config for teacher identity")?,
    )?;
    let executable_sha256 = validate_raw_sha256("server executable", executable_sha256)?;
    let numerical_runtime_sha256 =
        validate_raw_sha256("numerical runtime", numerical_runtime_sha256)?;
    let inference_config_sha256 = inference_config_sha256(
        model_config,
        backend,
        streaming_prefill,
        executable_sha256,
        numerical_runtime_sha256,
    )?;

    let vocab_size = u32::try_from(model_config.vocab_size)
        .context("model vocabulary size does not fit teacher identity")?;
    let max_model_len = u32::try_from(
        model_config
            .max_position_embeddings
            .min(MAX_COMPLETION_PROMPT_TOKENS),
    )
    .context("model context length does not fit teacher identity")?;
    let max_prompt_logprob_candidates = u32::try_from(MAX_COMPLETION_PROMPT_LOGPROB_CANDIDATES)
        .expect("prompt-logprob candidate cap fits u32");
    let max_top_k = u32::try_from(MAX_COMPLETION_PROMPT_LOGPROBS.min(model_config.vocab_size))
        .expect("prompt-logprobs hard cap fits u32");
    let implementation = format!(
        "kiln/{}/{}/binary-sha256:{}",
        env!("CARGO_PKG_VERSION"),
        backend,
        executable_sha256
    );

    TeacherIdentityV1::new(
        served_model_id,
        base_model_sha256,
        tokenizer_vocab_sha256,
        tokenizer_config_sha256,
        None,
        vocab_size,
        max_top_k,
        max_model_len,
        max_prompt_logprob_candidates,
        implementation,
        inference_config_sha256,
    )
    .context("invalid Kiln base teacher identity")
}

/// Bind a disk-loaded local LoRA to the exact base/runtime identity it will
/// score with. In-memory training views are rejected: they have no immutable
/// PEFT source revision until serialized and loaded through the production
/// loader.
pub fn build_local_adapter_teacher_identity(
    base_identity: &TeacherIdentityV1,
    adapter_name: &str,
    lora: &LoraWeights,
) -> Result<TeacherIdentityV1> {
    ensure!(
        base_identity.adapter().is_none(),
        "local adapter identity must derive from an unadapted base identity"
    );
    let source = lora
        .source_identity()
        .context("local adapter lacks loader-owned PEFT source identity")?;
    build_local_adapter_teacher_identity_from_source(base_identity, adapter_name, source)
}

/// Registration-time counterpart to [`build_local_adapter_teacher_identity`].
/// The job-start loader must derive the same source identity again before any
/// scores are accepted.
pub fn build_local_adapter_teacher_identity_from_source(
    base_identity: &TeacherIdentityV1,
    adapter_name: &str,
    source: &LoraSourceIdentity,
) -> Result<TeacherIdentityV1> {
    ensure!(
        base_identity.adapter().is_none(),
        "local adapter identity must derive from an unadapted base identity"
    );
    let adapter = TeacherAdapterIdentityV1::new(
        adapter_name,
        source.weights_sha256(),
        source.config_sha256(),
    )
    .context("invalid local adapter source identity")?;
    base_identity
        .with_static_adapter(adapter)
        .context("invalid local adapter teacher identity")
}

fn inference_config_sha256(
    model_config: &ModelConfig,
    backend: &str,
    streaming_prefill: StreamingPrefillExecutionPolicy,
    executable_sha256: &str,
    numerical_runtime_sha256: &str,
) -> Result<String> {
    let contract = KilnInferenceContract {
        schema: INFERENCE_CONTRACT_SCHEMA,
        model_config,
        backend,
        streaming_prefill: streaming_prefill.into(),
        executable_sha256,
        numerical_runtime_sha256,
        causal_alignment: CAUSAL_ALIGNMENT,
        logprob_semantics: LOGPROB_SEMANTICS,
        candidate_semantics: CANDIDATE_SEMANTICS,
        hard_caps: PromptLogprobHardCaps {
            max_top_k: MAX_COMPLETION_PROMPT_LOGPROBS,
            max_prompt_tokens: MAX_COMPLETION_PROMPT_TOKENS,
            max_response_candidates: MAX_COMPLETION_PROMPT_LOGPROB_CANDIDATES,
            projection_byte_budget: PROMPT_LOGPROB_PROJECTION_BYTE_BUDGET,
            max_projection_chunk_tokens: MAX_PROMPT_LOGPROB_PROJECTION_CHUNK_TOKENS,
        },
    };
    let bytes = serde_json::to_vec(&contract)
        .context("failed to serialize Kiln prompt-logprob inference contract")?;
    Ok(hex_sha256(&bytes))
}

/// Hash the exact executable inode running this process.
pub fn current_executable_sha256() -> Result<String> {
    #[cfg(target_os = "linux")]
    let path = std::path::PathBuf::from("/proc/self/exe");
    #[cfg(not(target_os = "linux"))]
    let path = std::env::current_exe().context("resolve current server executable")?;

    let mut file = std::fs::File::open(&path)
        .with_context(|| format!("open running executable {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hash running executable {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(bytes_to_hex(&digest.finalize()))
}

/// Hash the selected accelerator, driver/toolchain report, OS, and architecture.
/// Command output is never published directly; only this digest enters identity.
pub fn numerical_runtime_sha256(device: Device) -> String {
    let mut digest = Sha256::new();
    hash_framed(&mut digest, b"kiln.numerical-runtime.v2");
    hash_framed(&mut digest, std::env::consts::OS.as_bytes());
    hash_framed(&mut digest, std::env::consts::ARCH.as_bytes());
    hash_framed(&mut digest, device.short_name().as_bytes());
    hash_common_runtime_evidence(&mut digest);

    match device {
        Device::Cuda(index) => hash_command(
            &mut digest,
            "nvidia-smi",
            "nvidia-smi",
            &[
                "--query-gpu=name,uuid,driver_version,compute_cap",
                "--format=csv,noheader",
                "--id",
                &index.to_string(),
            ],
        ),
        Device::Rocm(_) => {
            hash_command(&mut digest, "rocminfo", "rocminfo", &[]);
            hash_command(
                &mut digest,
                "rocm-smi",
                "rocm-smi",
                &["--showproductname", "--showdriverversion"],
            );
        }
        Device::Vulkan(_) => {
            hash_command(&mut digest, "vulkaninfo", "vulkaninfo", &["--summary"]);
        }
        Device::Metal(_) => hash_command(
            &mut digest,
            "system-profiler-hardware-software-displays",
            "system_profiler",
            &[
                "SPHardwareDataType",
                "SPSoftwareDataType",
                "SPDisplaysDataType",
                "-json",
            ],
        ),
        Device::Cpu => hash_framed(&mut digest, b"cpu-runtime-evidence-complete"),
        _ => hash_framed(&mut digest, b"unknown-device-variant"),
    }
    bytes_to_hex(&digest.finalize())
}

fn hash_common_runtime_evidence(digest: &mut Sha256) {
    hash_framed(digest, b"common-runtime-evidence.v1");
    hash_framed(digest, runtime_isa_features().as_bytes());

    #[cfg(target_os = "linux")]
    {
        hash_runtime_file(digest, "os-release", Path::new("/etc/os-release"));
        hash_runtime_file(
            digest,
            "kernel-release",
            Path::new("/proc/sys/kernel/osrelease"),
        );
        let cpu = read_bounded_file(Path::new("/proc/cpuinfo"), RUNTIME_FILE_CAPTURE_BYTES);
        hash_framed(digest, b"cpuinfo-canonical");
        hash_capture_metadata(digest, &cpu);
        hash_framed(digest, canonical_cpu_identity(&cpu.bytes).as_bytes());
        let maps = read_bounded_file(Path::new("/proc/self/maps"), RUNTIME_FILE_CAPTURE_BYTES);
        hash_framed(digest, b"loaded-numerical-libraries");
        hash_capture_metadata(digest, &maps);
        hash_framed(
            digest,
            canonical_loaded_runtime_libraries(&maps.bytes).as_bytes(),
        );
        hash_command(digest, "uname", "uname", &["-a"]);
        hash_command(digest, "libc-version", "getconf", &["GNU_LIBC_VERSION"]);
    }

    #[cfg(target_os = "macos")]
    {
        hash_command(digest, "uname", "uname", &["-a"]);
        hash_command(digest, "os-build", "sw_vers", &[]);
        hash_command(
            digest,
            "cpu-hardware",
            "sysctl",
            &[
                "-n",
                "hw.model",
                "hw.machine",
                "machdep.cpu.brand_string",
                "machdep.cpu.features",
                "machdep.cpu.leaf7_features",
            ],
        );
    }

    #[cfg(target_os = "windows")]
    {
        hash_command(digest, "windows-version", "cmd", &["/C", "ver"]);
        hash_command(
            digest,
            "windows-cpu",
            "wmic",
            &["cpu", "get", "Name,Manufacturer,ProcessorId", "/value"],
        );
    }
}

fn hash_command(digest: &mut Sha256, label: &str, program: &str, args: &[&str]) {
    hash_framed(digest, label.as_bytes());
    hash_framed(digest, program.as_bytes());
    for arg in args {
        hash_framed(digest, arg.as_bytes());
    }
    hash_command_evidence(
        digest,
        &run_bounded_command(
            program,
            args,
            RUNTIME_COMMAND_TIMEOUT,
            RUNTIME_COMMAND_CAPTURE_BYTES,
        ),
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CommandOutcome {
    Exited(i32),
    TimedOut,
    SpawnError(String),
    WaitError(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct CapturedStream {
    bytes: Vec<u8>,
    truncated: bool,
    error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommandEvidence {
    outcome: CommandOutcome,
    stdout: CapturedStream,
    stderr: CapturedStream,
}

fn run_bounded_command(
    program: &str,
    args: &[&str],
    deadline: Duration,
    capture_limit: usize,
) -> CommandEvidence {
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return CommandEvidence {
                outcome: CommandOutcome::SpawnError(error.to_string()),
                stdout: CapturedStream::default(),
                stderr: CapturedStream::default(),
            };
        }
    };
    let stdout = child
        .stdout
        .take()
        .expect("piped child stdout is available");
    let stderr = child
        .stderr
        .take()
        .expect("piped child stderr is available");
    let stdout_reader = std::thread::spawn(move || read_stream_bounded(stdout, capture_limit));
    let stderr_reader = std::thread::spawn(move || read_stream_bounded(stderr, capture_limit));

    let started = Instant::now();
    let outcome = loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // A probe that exits after forking must not leave descendants
                // holding capture pipes open indefinitely.
                terminate_process_group(&mut child);
                break CommandOutcome::Exited(status.code().unwrap_or(-1));
            }
            Ok(None) if started.elapsed() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Ok(None) => {
                terminate_process_group(&mut child);
                let _ = child.wait();
                break CommandOutcome::TimedOut;
            }
            Err(error) => {
                terminate_process_group(&mut child);
                let _ = child.wait();
                break CommandOutcome::WaitError(error.to_string());
            }
        }
    };
    let stdout = stdout_reader.join().unwrap_or_else(|_| CapturedStream {
        error: Some("stdout capture thread panicked".to_string()),
        ..CapturedStream::default()
    });
    let stderr = stderr_reader.join().unwrap_or_else(|_| CapturedStream {
        error: Some("stderr capture thread panicked".to_string()),
        ..CapturedStream::default()
    });
    CommandEvidence {
        outcome,
        stdout,
        stderr,
    }
}

fn terminate_process_group(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        // SAFETY: the child was placed in a process group whose id equals its
        // pid. A negative pid targets that group and prevents descendants from
        // retaining output pipes past the deadline.
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
    }
    let _ = child.kill();
}

fn read_stream_bounded(mut reader: impl Read, capture_limit: usize) -> CapturedStream {
    let mut capture = CapturedStream::default();
    let mut buffer = [0u8; 16 * 1024];
    loop {
        match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => {
                let retain = capture_limit.saturating_sub(capture.bytes.len()).min(read);
                capture.bytes.extend_from_slice(&buffer[..retain]);
                capture.truncated |= retain < read;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                capture.error = Some(error.to_string());
                break;
            }
        }
    }
    capture
}

fn hash_command_evidence(digest: &mut Sha256, evidence: &CommandEvidence) {
    match &evidence.outcome {
        CommandOutcome::Exited(code) => {
            hash_framed(digest, b"exited");
            hash_framed(digest, &code.to_le_bytes());
        }
        CommandOutcome::TimedOut => hash_framed(digest, b"timed-out"),
        CommandOutcome::SpawnError(error) => {
            hash_framed(digest, b"spawn-error");
            hash_framed(digest, error.as_bytes());
        }
        CommandOutcome::WaitError(error) => {
            hash_framed(digest, b"wait-error");
            hash_framed(digest, error.as_bytes());
        }
    }
    hash_captured_stream(digest, &evidence.stdout);
    hash_captured_stream(digest, &evidence.stderr);
}

fn hash_captured_stream(digest: &mut Sha256, stream: &CapturedStream) {
    hash_capture_metadata(digest, stream);
    hash_framed(digest, stream.bytes.as_slice());
}

fn hash_capture_metadata(digest: &mut Sha256, stream: &CapturedStream) {
    hash_framed(digest, &[u8::from(stream.truncated)]);
    hash_framed(
        digest,
        stream.error.as_deref().unwrap_or_default().as_bytes(),
    );
}

fn read_bounded_file(path: &Path, limit: usize) -> CapturedStream {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) => {
            return CapturedStream {
                error: Some(error.to_string()),
                ..CapturedStream::default()
            };
        }
    };
    read_stream_bounded(file, limit)
}

fn hash_runtime_file(digest: &mut Sha256, label: &str, path: &Path) {
    hash_framed(digest, label.as_bytes());
    hash_framed(digest, path.as_os_str().to_string_lossy().as_bytes());
    hash_captured_stream(digest, &read_bounded_file(path, RUNTIME_FILE_CAPTURE_BYTES));
}

fn canonical_cpu_identity(input: &[u8]) -> String {
    const KEYS: &[&str] = &[
        "vendor_id",
        "model name",
        "cpu family",
        "model",
        "stepping",
        "microcode",
        "flags",
        "features",
        "cpu implementer",
        "cpu architecture",
        "cpu variant",
        "cpu part",
        "cpu revision",
        "hardware",
    ];
    let text = String::from_utf8_lossy(input);
    let mut records = BTreeSet::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        if !KEYS.contains(&key.as_str()) {
            continue;
        }
        let value = if matches!(key.as_str(), "flags" | "features") {
            let features = value.split_whitespace().collect::<BTreeSet<_>>();
            features.into_iter().collect::<Vec<_>>().join(" ")
        } else {
            value.split_whitespace().collect::<Vec<_>>().join(" ")
        };
        records.insert(format!("{key}={value}"));
    }
    records.into_iter().collect::<Vec<_>>().join("\n")
}

fn canonical_loaded_runtime_libraries(input: &[u8]) -> String {
    const NEEDLES: &[&str] = &[
        "libamdhip64",
        "libcuda",
        "libcudart",
        "libvulkan",
        "libmetal",
        "libc.so",
        "libm.so",
        "libstdc++",
        "libsystem",
    ];
    let text = String::from_utf8_lossy(input);
    let mut records = BTreeSet::new();
    for line in text.lines() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 6 {
            continue;
        }
        let path = fields[5..].join(" ");
        let normalized = path.to_ascii_lowercase();
        if NEEDLES.iter().any(|needle| normalized.contains(needle)) {
            records.insert(format!("{}|{}|{path}", fields[3], fields[4]));
        }
    }
    records.into_iter().collect::<Vec<_>>().join("\n")
}

fn runtime_isa_features() -> String {
    let mut features = Vec::new();
    #[cfg(target_arch = "x86_64")]
    for (name, enabled) in [
        ("sse2", std::is_x86_feature_detected!("sse2")),
        ("sse4.1", std::is_x86_feature_detected!("sse4.1")),
        ("avx", std::is_x86_feature_detected!("avx")),
        ("avx2", std::is_x86_feature_detected!("avx2")),
        ("fma", std::is_x86_feature_detected!("fma")),
        ("avx512f", std::is_x86_feature_detected!("avx512f")),
        ("avx512bf16", std::is_x86_feature_detected!("avx512bf16")),
    ] {
        if enabled {
            features.push(name);
        }
    }
    #[cfg(target_arch = "aarch64")]
    for (name, enabled) in [
        ("neon", std::arch::is_aarch64_feature_detected!("neon")),
        ("fp", std::arch::is_aarch64_feature_detected!("fp")),
        ("fp16", std::arch::is_aarch64_feature_detected!("fp16")),
        ("sve", std::arch::is_aarch64_feature_detected!("sve")),
    ] {
        if enabled {
            features.push(name);
        }
    }
    features.sort_unstable();
    features.join(",")
}

fn hash_framed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn validate_raw_sha256<'a>(field: &str, value: &'a str) -> Result<&'a str> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{field} SHA-256 must contain exactly 64 lowercase hexadecimal characters"
    );
    Ok(value)
}

fn strip_sha256_prefix(field: &str, value: &str) -> Result<String> {
    let digest = value
        .strip_prefix("sha256:")
        .with_context(|| format!("{field} must use the `sha256:` prefix"))?;
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{field} must contain exactly 64 lowercase hexadecimal characters"
    );
    Ok(digest.to_owned())
}

fn hex_sha256(bytes: &[u8]) -> String {
    bytes_to_hex(&Sha256::digest(bytes))
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXECUTABLE_HASH: &str =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const RUNTIME_HASH: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn tokenizer() -> KilnTokenizer {
        let vocab = (0..512u32)
            .map(|token_id| (format!("token-{token_id}"), token_id))
            .collect::<std::collections::HashMap<_, _>>();
        KilnTokenizer::from_bytes(
            &serde_json::to_vec(&serde_json::json!({
                "version": "1.0",
                "model": { "type": "BPE", "vocab": vocab, "merges": [] }
            }))
            .unwrap(),
        )
        .unwrap()
    }

    fn model_config() -> ModelConfig {
        let mut config = ModelConfig::qwen3_5_4b();
        config.vocab_size = 512;
        config.max_position_embeddings = 4096;
        config
    }

    fn streaming_prefill_policy() -> StreamingPrefillExecutionPolicy {
        StreamingPrefillExecutionPolicy::for_device(Device::Cpu)
    }

    fn base_identity() -> TeacherIdentityV1 {
        build_base_teacher_identity(
            "kiln-test",
            &format!("sha256:{}", "a".repeat(64)),
            &tokenizer(),
            &model_config(),
            "cpu",
            streaming_prefill_policy(),
            EXECUTABLE_HASH,
            RUNTIME_HASH,
        )
        .unwrap()
    }

    #[test]
    fn constructs_complete_base_identity() {
        let tokenizer = tokenizer();
        let identity = build_base_teacher_identity(
            "kiln-test",
            &format!("sha256:{}", "a".repeat(64)),
            &tokenizer,
            &model_config(),
            "vulkan",
            streaming_prefill_policy(),
            EXECUTABLE_HASH,
            RUNTIME_HASH,
        )
        .unwrap();

        assert_eq!(identity.served_model_id(), "kiln-test");
        assert_eq!(identity.base_model_sha256(), "a".repeat(64));
        assert_eq!(identity.vocab_size(), 512);
        assert_eq!(identity.max_top_k(), 256);
        assert_eq!(identity.max_model_len(), 4096);
        assert_eq!(identity.max_prompt_logprob_candidates(), 65_536);
        assert!(identity.adapter().is_none());
        assert_eq!(
            identity.implementation(),
            format!(
                "kiln/{}/vulkan/binary-sha256:{EXECUTABLE_HASH}",
                env!("CARGO_PKG_VERSION")
            )
        );
        assert_eq!(
            TeacherIdentityV1::parse_fingerprint(&identity.fingerprint()).unwrap(),
            identity
        );
    }

    #[test]
    fn constructs_identity_with_padded_model_vocabulary() {
        let mut config = model_config();
        config.vocab_size = 768;
        let identity = build_base_teacher_identity(
            "kiln-test",
            &format!("sha256:{}", "a".repeat(64)),
            &tokenizer(),
            &config,
            "cpu",
            streaming_prefill_policy(),
            EXECUTABLE_HASH,
            RUNTIME_HASH,
        )
        .unwrap();

        assert_eq!(identity.vocab_size(), 768);
        assert_eq!(identity.max_top_k(), 256);
    }

    #[test]
    fn base_identity_is_bound_to_the_exact_streaming_prefill_policy() {
        let tokenizer = tokenizer();
        let default_policy = streaming_prefill_policy();
        let forced_streaming_policy = StreamingPrefillExecutionPolicy::resolve(
            kiln_model::StreamingPrefillBackendPolicy::for_device(Device::Cpu),
            StreamingPrefillMode::Enabled,
            Some(17),
            Some(19),
            Some(23),
            Some(29),
            false,
        );
        let build = |policy| {
            build_base_teacher_identity(
                "kiln-test",
                &format!("sha256:{}", "a".repeat(64)),
                &tokenizer,
                &model_config(),
                "cpu",
                policy,
                EXECUTABLE_HASH,
                RUNTIME_HASH,
            )
            .unwrap()
        };

        let default_identity = build(default_policy);
        let configured_identity = build(forced_streaming_policy);
        assert_ne!(
            default_identity.inference_config_sha256(),
            configured_identity.inference_config_sha256()
        );
        assert_ne!(
            default_identity.content_revision(),
            configured_identity.content_revision()
        );
    }

    #[test]
    fn local_adapter_identity_uses_exact_loader_owned_source() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("adapter_config.json"),
            br#"{"r":1,"lora_alpha":1.0,"target_modules":[]}"#,
        )
        .unwrap();
        let tensor_bytes = 1.0f32.to_le_bytes();
        let tensor =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &tensor_bytes)
                .unwrap();
        let serialized =
            safetensors::tensor::serialize([("ignored.weight", tensor)].into_iter(), None).unwrap();
        std::fs::write(dir.path().join("adapter_model.safetensors"), serialized).unwrap();

        let lora = LoraWeights::load(dir.path(), 0, Device::Cpu).unwrap();
        let source = lora.source_identity().unwrap();
        let base = base_identity();
        let identity = build_local_adapter_teacher_identity(&base, "prior-self", &lora).unwrap();
        let adapter = identity.adapter().unwrap();

        assert_eq!(adapter.name(), "prior-self");
        assert_eq!(adapter.weights_sha256(), source.weights_sha256());
        assert_eq!(adapter.config_sha256(), source.config_sha256());
        assert_eq!(identity.base_model_sha256(), base.base_model_sha256());
        assert_eq!(
            identity.inference_config_sha256(),
            base.inference_config_sha256()
        );
        assert_ne!(identity.content_revision(), base.content_revision());
    }

    #[test]
    fn local_adapter_identity_rejects_unserialized_tensor_views() {
        let lora = LoraWeights {
            layers: Vec::new(),
            mtp: None,
            rank: 1,
            alpha: 1.0,
            scale: 1.0,
            source_identity: None,
        };
        let error =
            build_local_adapter_teacher_identity(&base_identity(), "memory", &lora).unwrap_err();
        assert!(
            error.to_string().contains("lacks loader-owned"),
            "{error:#}"
        );
    }

    #[test]
    fn rejects_tokenizer_entry_count_exceeding_model_vocabulary() {
        let mut config = model_config();
        config.vocab_size = 511;
        let error = build_base_teacher_identity(
            "kiln-test",
            &format!("sha256:{}", "a".repeat(64)),
            &tokenizer(),
            &config,
            "cpu",
            streaming_prefill_policy(),
            EXECUTABLE_HASH,
            RUNTIME_HASH,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("entry count 512 exceeds model vocabulary size 511")
        );
    }

    #[test]
    fn rejects_tokenizer_id_outside_model_vocabulary() {
        let tokenizer = KilnTokenizer::from_bytes(
            br#"{
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab": {"token-0": 0, "token-512": 512},
                    "merges": []
                }
            }"#,
        )
        .unwrap();
        let error = build_base_teacher_identity(
            "kiln-test",
            &format!("sha256:{}", "a".repeat(64)),
            &tokenizer,
            &model_config(),
            "cpu",
            streaming_prefill_policy(),
            EXECUTABLE_HASH,
            RUNTIME_HASH,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("maximum token ID 512 is outside model vocabulary size 512")
        );
    }

    #[test]
    fn rejects_malformed_loader_hashes() {
        for malformed in [
            "a".repeat(64),
            format!("sha256:{}", "a".repeat(63)),
            format!("sha256:{}", "A".repeat(64)),
            format!("sha256:{}g", "a".repeat(63)),
        ] {
            let error = build_base_teacher_identity(
                "kiln-test",
                &malformed,
                &tokenizer(),
                &model_config(),
                "cpu",
                streaming_prefill_policy(),
                EXECUTABLE_HASH,
                RUNTIME_HASH,
            )
            .unwrap_err();
            assert!(
                error.to_string().contains("loader-owned base model"),
                "unexpected error for {malformed:?}: {error:#}"
            );
        }
    }

    #[test]
    fn rejects_missing_backend_identity() {
        let error = build_base_teacher_identity(
            "kiln-test",
            &format!("sha256:{}", "a".repeat(64)),
            &tokenizer(),
            &model_config(),
            "",
            streaming_prefill_policy(),
            EXECUTABLE_HASH,
            RUNTIME_HASH,
        )
        .unwrap_err();
        assert!(error.to_string().contains("backend name"));
    }

    #[test]
    fn inference_contract_is_stable_and_backend_bound() {
        let config = model_config();
        let streaming_prefill = streaming_prefill_policy();
        let cpu = inference_config_sha256(
            &config,
            "cpu",
            streaming_prefill,
            EXECUTABLE_HASH,
            RUNTIME_HASH,
        )
        .unwrap();
        assert_eq!(
            cpu,
            inference_config_sha256(
                &config,
                "cpu",
                streaming_prefill,
                EXECUTABLE_HASH,
                RUNTIME_HASH,
            )
            .unwrap()
        );
        assert_ne!(
            cpu,
            inference_config_sha256(
                &config,
                "vulkan",
                streaming_prefill,
                EXECUTABLE_HASH,
                RUNTIME_HASH,
            )
            .unwrap()
        );
        assert_ne!(
            cpu,
            inference_config_sha256(
                &config,
                "cpu",
                streaming_prefill,
                &"f".repeat(64),
                RUNTIME_HASH,
            )
            .unwrap()
        );
        assert_ne!(
            cpu,
            inference_config_sha256(
                &config,
                "cpu",
                streaming_prefill,
                EXECUTABLE_HASH,
                &"f".repeat(64),
            )
            .unwrap()
        );

        let mut different_context = config;
        different_context.max_position_embeddings += 1;
        assert_ne!(
            cpu,
            inference_config_sha256(
                &different_context,
                "cpu",
                streaming_prefill,
                EXECUTABLE_HASH,
                RUNTIME_HASH,
            )
            .unwrap()
        );
    }

    #[test]
    fn inference_contract_binds_every_streaming_prefill_policy_field() {
        let config = model_config();
        let backend_policy = kiln_model::StreamingPrefillBackendPolicy {
            auto_dispatch: kiln_model::StreamingPrefillAutoDispatch::PromptTokensAtLeast(101),
            base_tile_tokens: 103,
            tape_tile_tokens: 107,
            detached_full_attn_tile_tokens: 109,
            detached_full_attn_boundary_tile_tokens: 113,
            detached_full_attn_tape_replay_tile_tokens: 127,
        };
        let hash = |policy| {
            inference_config_sha256(&config, "cpu", policy, EXECUTABLE_HASH, RUNTIME_HASH).unwrap()
        };
        let baseline_policy = StreamingPrefillExecutionPolicy::from_backend_policy(backend_policy);
        let baseline = hash(baseline_policy);
        let distinct_policies = [
            StreamingPrefillExecutionPolicy::resolve(
                backend_policy,
                StreamingPrefillMode::Disabled,
                None,
                None,
                None,
                None,
                true,
            ),
            StreamingPrefillExecutionPolicy::from_backend_policy(
                kiln_model::StreamingPrefillBackendPolicy {
                    auto_dispatch: kiln_model::StreamingPrefillAutoDispatch::PromptTokensAtLeast(
                        131,
                    ),
                    ..backend_policy
                },
            ),
            StreamingPrefillExecutionPolicy::from_backend_policy(
                kiln_model::StreamingPrefillBackendPolicy {
                    base_tile_tokens: 137,
                    ..backend_policy
                },
            ),
            StreamingPrefillExecutionPolicy::from_backend_policy(
                kiln_model::StreamingPrefillBackendPolicy {
                    tape_tile_tokens: 139,
                    ..backend_policy
                },
            ),
            StreamingPrefillExecutionPolicy::from_backend_policy(
                kiln_model::StreamingPrefillBackendPolicy {
                    detached_full_attn_tile_tokens: 149,
                    ..backend_policy
                },
            ),
            StreamingPrefillExecutionPolicy::from_backend_policy(
                kiln_model::StreamingPrefillBackendPolicy {
                    detached_full_attn_boundary_tile_tokens: 151,
                    ..backend_policy
                },
            ),
            StreamingPrefillExecutionPolicy::from_backend_policy(
                kiln_model::StreamingPrefillBackendPolicy {
                    detached_full_attn_tape_replay_tile_tokens: 157,
                    ..backend_policy
                },
            ),
            StreamingPrefillExecutionPolicy::resolve(
                backend_policy,
                StreamingPrefillMode::Auto,
                None,
                None,
                None,
                None,
                false,
            ),
        ];

        for policy in distinct_policies {
            assert_ne!(
                baseline,
                hash(policy),
                "policy difference was not identity-bound"
            );
        }
    }

    #[test]
    fn running_executable_and_runtime_hashes_are_canonical() {
        let executable = current_executable_sha256().unwrap();
        assert!(validate_raw_sha256("test executable", &executable).is_ok());
        let runtime = numerical_runtime_sha256(Device::Cpu);
        assert!(validate_raw_sha256("test runtime", &runtime).is_ok());
        assert_eq!(runtime, numerical_runtime_sha256(Device::Cpu));
    }

    #[test]
    fn cpu_identity_parser_ignores_frequency_and_core_order() {
        let first = br#"
processor : 0
vendor_id : AuthenticAMD
model name : Example CPU
microcode : 0x123
cpu MHz : 900.0
flags : avx2 sse2 fma
processor : 1
vendor_id : AuthenticAMD
model name : Example CPU
microcode : 0x123
cpu MHz : 4100.0
flags : fma avx2 sse2
"#;
        let second = br#"
processor : 9
model name: Example   CPU
vendor_id: AuthenticAMD
flags: sse2 fma avx2
microcode: 0x123
cpu MHz: 1200.0
"#;
        assert_eq!(
            canonical_cpu_identity(first),
            canonical_cpu_identity(second)
        );
        assert!(canonical_cpu_identity(first).contains("microcode=0x123"));
    }

    #[test]
    fn loaded_library_parser_drops_address_aslr_but_binds_inode_and_path() {
        let first = b"7f00-7f10 r-xp 0000 08:01 42 /usr/lib/libvulkan.so.1\n";
        let relocated = b"9a00-9a10 r-xp 0000 08:01 42 /usr/lib/libvulkan.so.1\n";
        let replaced = b"9a00-9a10 r-xp 0000 08:01 43 /usr/lib/libvulkan.so.1\n";
        assert_eq!(
            canonical_loaded_runtime_libraries(first),
            canonical_loaded_runtime_libraries(relocated)
        );
        assert_ne!(
            canonical_loaded_runtime_libraries(first),
            canonical_loaded_runtime_libraries(replaced)
        );
    }

    #[test]
    fn command_evidence_framing_binds_status_output_truncation_and_errors() {
        fn digest(evidence: &CommandEvidence) -> String {
            let mut hash = Sha256::new();
            hash_command_evidence(&mut hash, evidence);
            bytes_to_hex(&hash.finalize())
        }
        let base = CommandEvidence {
            outcome: CommandOutcome::Exited(0),
            stdout: CapturedStream {
                bytes: b"runtime-v1".to_vec(),
                truncated: false,
                error: None,
            },
            stderr: CapturedStream::default(),
        };
        assert_eq!(digest(&base), digest(&base));
        let mut changed = base.clone();
        changed.stdout.truncated = true;
        assert_ne!(digest(&base), digest(&changed));
        changed = base.clone();
        changed.outcome = CommandOutcome::TimedOut;
        assert_ne!(digest(&base), digest(&changed));
        changed = base.clone();
        changed.stdout.error = Some("read failed".into());
        assert_ne!(digest(&base), digest(&changed));
    }

    #[cfg(unix)]
    #[test]
    fn runtime_command_capture_enforces_deadline_and_output_cap() {
        let timed_out =
            run_bounded_command("sh", &["-c", "sleep 2"], Duration::from_millis(50), 128);
        assert_eq!(timed_out.outcome, CommandOutcome::TimedOut);

        let capped = run_bounded_command(
            "sh",
            &["-c", "printf '%02048d' 0"],
            Duration::from_secs(1),
            64,
        );
        assert_eq!(capped.outcome, CommandOutcome::Exited(0));
        assert_eq!(capped.stdout.bytes.len(), 64);
        assert!(capped.stdout.truncated);

        let started = Instant::now();
        let descendant = run_bounded_command(
            "sh",
            &["-c", "(sleep 10) & exit 0"],
            Duration::from_secs(1),
            128,
        );
        assert_eq!(descendant.outcome, CommandOutcome::Exited(0));
        assert!(started.elapsed() < Duration::from_secs(2));
    }
}
