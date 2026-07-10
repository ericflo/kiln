use std::fmt;
use std::fs::File;
use std::io;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use sha2::{Digest, Sha256};

/// Data type for stored tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F16,
    BF16,
    F32,
}

impl TensorDType {
    /// Bytes per element.
    pub fn size_bytes(self) -> usize {
        match self {
            TensorDType::F16 | TensorDType::BF16 => 2,
            TensorDType::F32 => 4,
        }
    }
}

impl fmt::Display for TensorDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorDType::F16 => write!(f, "f16"),
            TensorDType::BF16 => write!(f, "bf16"),
            TensorDType::F32 => write!(f, "f32"),
        }
    }
}

/// Backing storage for loaded tensor bytes.
#[derive(Clone)]
pub enum WeightData {
    /// Owned bytes, used by generated/dequantized tensors and tests.
    Owned(Vec<u8>),
    /// Read-only slice into a memory-mapped safetensors shard.
    MmapSlice {
        mmap: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
}

impl WeightData {
    pub fn owned(data: Vec<u8>) -> Self {
        Self::Owned(data)
    }

    pub fn mmap_slice(mmap: Arc<Mmap>, offset: usize, len: usize) -> Self {
        Self::MmapSlice { mmap, offset, len }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            WeightData::Owned(data) => data,
            WeightData::MmapSlice { mmap, offset, len } => &mmap[*offset..*offset + *len],
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        self.as_bytes()
    }

    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }
}

impl Deref for WeightData {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_bytes()
    }
}

impl fmt::Debug for WeightData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightData::Owned(data) => f.debug_struct("Owned").field("len", &data.len()).finish(),
            WeightData::MmapSlice { offset, len, .. } => f
                .debug_struct("MmapSlice")
                .field("offset", offset)
                .field("len", len)
                .finish(),
        }
    }
}

/// Provenance for a loaded weight tensor.
///
/// This is used by the persistent transpose cache to key entries by the exact
/// checkpoint shard that supplied the bytes.
#[derive(Debug, Clone)]
pub struct WeightSource {
    pub shard_path: PathBuf,
    pub shard_size: u64,
    pub shard_mtime_ns: u128,
    pub tensor_name: String,
}

/// A loaded tensor: raw bytes with shape and dtype metadata.
///
/// This is a CPU-side representation. The forward pass will convert these
/// to GPU tensors (candle Tensor or raw CUDA buffers).
#[derive(Clone)]
pub struct WeightTensor {
    pub data: WeightData,
    pub shape: Vec<usize>,
    pub dtype: TensorDType,
    pub source: Option<WeightSource>,
}

impl WeightTensor {
    /// Total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Raw tensor bytes in row-major safetensors order.
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }
}

impl fmt::Debug for WeightTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ds = f.debug_struct("WeightTensor");
        ds.field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("bytes", &self.data.len());
        if let Some(source) = &self.source {
            ds.field("source", source);
        }
        ds.finish()
    }
}

/// Token embedding weights.
#[derive(Debug, Clone)]
pub struct EmbeddingWeights {
    /// [vocab_size, hidden_size]
    pub embed_tokens: WeightTensor,
}

/// Standard GQA self-attention weights (for full attention layers).
///
/// These layers use FlashAttention with KV cache and RoPE.
/// Every 4th layer in Qwen3.5-4B (indices 3, 7, 11, ..., 31).
#[derive(Debug, Clone)]
pub struct FullAttentionWeights {
    /// [num_heads * head_dim, hidden_size]
    pub q_proj: WeightTensor,
    /// [num_kv_heads * head_dim, hidden_size]
    pub k_proj: WeightTensor,
    /// [num_kv_heads * head_dim, hidden_size]
    pub v_proj: WeightTensor,
    /// [hidden_size, num_heads * head_dim]
    pub o_proj: WeightTensor,
    /// QK normalization weights (RMSNorm per head).
    /// [head_dim]
    pub q_norm: WeightTensor,
    /// [head_dim]
    pub k_norm: WeightTensor,
}

/// Gated DeltaNet linear attention weights.
///
/// These layers use O(1) recurrent state instead of KV cache.
/// 24 out of 32 layers in Qwen3.5-4B use this mechanism.
#[derive(Debug, Clone)]
pub struct LinearAttentionWeights {
    /// Fused QKV projection. [3 * num_heads * head_dim, hidden_size]
    pub in_proj_qkv: WeightTensor,
    /// Gate projection for output gating. [num_heads * head_dim, hidden_size]
    pub in_proj_z: WeightTensor,
    /// Output projection. [hidden_size, num_heads * head_dim]
    pub out_proj: WeightTensor,
    /// Alpha gate input projection. [num_heads * head_dim, hidden_size]
    pub in_proj_a: WeightTensor,
    /// Beta gate input projection. [num_heads * head_dim, hidden_size]
    pub in_proj_b: WeightTensor,
    /// Short convolution (causal conv1d). [num_heads * head_dim, 1, conv_size]
    pub conv1d: WeightTensor,
    /// Group norm weights. [num_heads * head_dim]
    pub norm: WeightTensor,
    /// Log of the A matrix (discretization parameter). [num_heads * head_dim]
    pub a_log: WeightTensor,
    /// Time-step bias. [num_heads * head_dim]
    pub dt_bias: WeightTensor,
}

/// Attention weights — either full GQA or linear (Gated DeltaNet).
#[derive(Debug, Clone)]
pub enum AttentionWeights {
    Full(FullAttentionWeights),
    Linear(LinearAttentionWeights),
}

/// SwiGLU feed-forward network weights (shared across all layer types).
#[derive(Debug, Clone)]
pub struct FfnWeights {
    /// [intermediate_size, hidden_size]
    pub gate_proj: WeightTensor,
    /// [intermediate_size, hidden_size]
    pub up_proj: WeightTensor,
    /// [hidden_size, intermediate_size]
    pub down_proj: WeightTensor,
}

/// One transformer layer's complete weights.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// RMSNorm before attention. [hidden_size]
    pub input_layernorm: WeightTensor,
    /// RMSNorm before FFN. [hidden_size]
    pub post_attention_layernorm: WeightTensor,
    /// Attention weights (full or linear depending on layer index).
    pub attention: AttentionWeights,
    /// Feed-forward network weights.
    pub mlp: FfnWeights,
}

/// Native MTP (Multi-Token Prediction) head weights for Qwen3.5-4B.
///
/// The pretrained Qwen3.5-4B checkpoint ships 15 MTP-prefixed tensors
/// (`num_nextn_predict_layers = 1` → `k=1` draft depth). The MTP head
/// lets us draft one token per decode step using the model's own
/// distilled head instead of a skip-layer self-spec approximation.
///
/// Forward shape (vLLM `qwen3_next_mtp.py` reference):
/// `concat(pre_fc_norm_embedding(embed(t)), pre_fc_norm_hidden(h)) → fc (2H→H)
///  → layer (GQA + SwiGLU MLP) → final_layernorm → tied lm_head (= embed_tokens.t())`
///
/// `lm_head` is tied to the base model's `embed_tokens`, so we do NOT
/// store a separate `lm_head` tensor — the spec-decode forward path
/// reuses `GpuWeights::embed_tokens_t`.
#[derive(Debug, Clone)]
pub struct MtpWeights {
    /// Concat-then-project: `[hidden_size, 2 * hidden_size]`.
    /// Ingests `concat(norm_embed, norm_hidden)` and produces `[seq, hidden_size]`.
    pub fc: WeightTensor,
    /// RMSNorm applied to the draft-candidate's token embedding before concat. `[hidden_size]`.
    pub pre_fc_norm_embedding: WeightTensor,
    /// RMSNorm applied to the base model's last hidden state before concat. `[hidden_size]`.
    pub pre_fc_norm_hidden: WeightTensor,
    /// Single MTP transformer layer (full GQA attention + SwiGLU MLP + input/post
    /// layernorms). Shape matches the main model's full-attention layer.
    pub layer: LayerWeights,
    /// Final RMSNorm before the tied lm_head. `[hidden_size]`.
    pub final_layernorm: WeightTensor,
}

impl MtpWeights {
    /// Total size of all MTP tensors in bytes.
    pub fn total_bytes(&self) -> usize {
        let mut total = self.fc.size_bytes();
        total += self.pre_fc_norm_embedding.size_bytes();
        total += self.pre_fc_norm_hidden.size_bytes();
        total += self.final_layernorm.size_bytes();
        total += self.layer.input_layernorm.size_bytes();
        total += self.layer.post_attention_layernorm.size_bytes();
        total += self.layer.mlp.gate_proj.size_bytes();
        total += self.layer.mlp.up_proj.size_bytes();
        total += self.layer.mlp.down_proj.size_bytes();
        match &self.layer.attention {
            AttentionWeights::Full(attn) => {
                total += attn.q_proj.size_bytes();
                total += attn.k_proj.size_bytes();
                total += attn.v_proj.size_bytes();
                total += attn.o_proj.size_bytes();
                total += attn.q_norm.size_bytes();
                total += attn.k_norm.size_bytes();
            }
            AttentionWeights::Linear(attn) => {
                total += attn.in_proj_qkv.size_bytes();
                total += attn.in_proj_z.size_bytes();
                total += attn.out_proj.size_bytes();
                total += attn.in_proj_a.size_bytes();
                total += attn.in_proj_b.size_bytes();
                total += attn.conv1d.size_bytes();
                total += attn.norm.size_bytes();
                total += attn.a_log.size_bytes();
                total += attn.dt_bias.size_bytes();
            }
        }
        total
    }

    /// Total parameter count across all MTP tensors.
    pub fn total_params(&self) -> usize {
        let mut total = self.fc.numel();
        total += self.pre_fc_norm_embedding.numel();
        total += self.pre_fc_norm_hidden.numel();
        total += self.final_layernorm.numel();
        total += self.layer.input_layernorm.numel();
        total += self.layer.post_attention_layernorm.numel();
        total += self.layer.mlp.gate_proj.numel();
        total += self.layer.mlp.up_proj.numel();
        total += self.layer.mlp.down_proj.numel();
        match &self.layer.attention {
            AttentionWeights::Full(attn) => {
                total += attn.q_proj.numel();
                total += attn.k_proj.numel();
                total += attn.v_proj.numel();
                total += attn.o_proj.numel();
                total += attn.q_norm.numel();
                total += attn.k_norm.numel();
            }
            AttentionWeights::Linear(attn) => {
                total += attn.in_proj_qkv.numel();
                total += attn.in_proj_z.numel();
                total += attn.out_proj.numel();
                total += attn.in_proj_a.numel();
                total += attn.in_proj_b.numel();
                total += attn.conv1d.numel();
                total += attn.norm.numel();
                total += attn.a_log.numel();
                total += attn.dt_bias.numel();
            }
        }
        total
    }
}

/// Deferred native-MTP source for startup-sensitive callers.
///
/// Keeps enough information to reopen the checkpoint and load only the
/// `mtp.*` tensors on demand later, instead of materializing them into
/// `ModelWeights` during process startup.
#[derive(Debug, Clone)]
pub struct DeferredMtpSource {
    pub model_dir: PathBuf,
    pub mtp_prefix: String,
    pub config: kiln_core::config::ModelConfig,
    /// Keeps the private immutable checkpoint snapshot alive until lazy MTP
    /// materialization has either completed or the runner is dropped.
    pub(crate) _snapshot_lease: Arc<ModelSnapshotLease>,
}

/// Lifetime owner for a private model snapshot. The directory is deleted when
/// the last CPU source guard or deferred-MTP source releases it.
#[derive(Debug)]
pub(crate) struct ModelSnapshotLease {
    pub(crate) directory: tempfile::TempDir,
}

/// Loader-owned proof that the bytes used to construct CPU weights remain the
/// bytes named by the model revision.
///
/// Keeping `Arc<Mmap>` values pins every mapping, including shards from which
/// no language-model tensor was extracted. The matching open files let the
/// post-upload check use bounded reads instead of touching a mapping after its
/// backing file may have been truncated (which can raise SIGBUS on Unix).
#[derive(Debug)]
pub(crate) struct SourceContentGuard {
    shards: Vec<SourceContentShard>,
    initial_shard_count: usize,
    initial_sha256: String,
    /// Declared after retained files so Windows closes mappings/handles before
    /// `TempDir` attempts recursive cleanup.
    _snapshot_lease: Arc<ModelSnapshotLease>,
}

#[derive(Debug)]
struct SourceContentShard {
    file: Arc<File>,
    mmap: Arc<Mmap>,
}

impl SourceContentGuard {
    pub(crate) fn new(
        shards: Vec<(Arc<File>, Arc<Mmap>)>,
        snapshot_lease: Arc<ModelSnapshotLease>,
    ) -> Self {
        let shards = shards
            .into_iter()
            .map(|(file, mmap)| SourceContentShard { file, mmap })
            .collect::<Vec<_>>();
        let initial_shard_count = shards.len();
        let records = shards
            .iter()
            .map(|shard| {
                let digest: [u8; 32] = Sha256::digest(&shard.mmap[..]).into();
                (digest, shard.mmap.len() as u64)
            })
            .collect();
        let initial_sha256 = source_content_sha256_from_records(records);
        Self {
            shards,
            initial_shard_count,
            initial_sha256,
            _snapshot_lease: snapshot_lease,
        }
    }

    pub(crate) fn initial_sha256(&self) -> &str {
        &self.initial_sha256
    }

    #[cfg(test)]
    pub(crate) fn snapshot_root(&self) -> &std::path::Path {
        self._snapshot_lease.directory.path()
    }

    fn verify_unchanged(&self) -> Result<()> {
        if self.shards.len() != self.initial_shard_count {
            bail!(
                "model source shard count changed after load: expected {}, observed {}",
                self.initial_shard_count,
                self.shards.len()
            );
        }

        let mut records = Vec::with_capacity(self.shards.len());
        for (index, shard) in self.shards.iter().enumerate() {
            let expected_len = shard.mmap.len() as u64;
            let before_len = shard
                .file
                .metadata()
                .with_context(|| format!("failed to stat retained model source shard {index}"))?
                .len();
            if before_len != expected_len {
                bail!(
                    "model source shard {index} length changed after load: expected {expected_len} bytes, observed {before_len}"
                );
            }

            let digest = hash_open_file_exact(&shard.file, expected_len)
                .with_context(|| format!("failed to verify model source shard {index}"))?;
            let after_len = shard
                .file
                .metadata()
                .with_context(|| format!("failed to restat retained model source shard {index}"))?
                .len();
            if after_len != expected_len {
                bail!(
                    "model source shard {index} length changed during verification: expected {expected_len} bytes, observed {after_len}"
                );
            }
            records.push((digest, expected_len));
        }

        let observed = source_content_sha256_from_records(records);
        if observed != self.initial_sha256 {
            bail!(
                "model source content changed after load: expected {}, observed {observed}",
                self.initial_sha256
            );
        }
        Ok(())
    }
}

const SOURCE_VERIFY_BUFFER_BYTES: usize = 256 * 1024;

fn hash_open_file_exact(file: &File, expected_len: u64) -> Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; SOURCE_VERIFY_BUFFER_BYTES];
    let mut offset = 0u64;
    while offset < expected_len {
        let remaining = usize::try_from((expected_len - offset).min(buffer.len() as u64))
            .expect("bounded source verification read length must fit usize");
        let read = loop {
            match read_file_at(file, &mut buffer[..remaining], offset) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                result => break result,
            }
        }?;
        if read == 0 {
            bail!("model source ended at byte {offset}, before the expected {expected_len} bytes");
        }
        hasher.update(&buffer[..read]);
        offset += read as u64;
    }

    let mut extra = [0u8; 1];
    if read_file_at(file, &mut extra, expected_len)? != 0 {
        bail!("model source grew beyond the expected {expected_len} bytes");
    }
    Ok(hasher.finalize().into())
}

#[cfg(unix)]
fn read_file_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buffer, offset)
}

#[cfg(windows)]
fn read_file_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buffer, offset)
}

#[cfg(not(any(unix, windows)))]
fn read_file_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::io::{Read, Seek, SeekFrom};

    let mut clone = file.try_clone()?;
    clone.seek(SeekFrom::Start(offset))?;
    clone.read(buffer)
}

fn source_content_sha256_from_records(mut records: Vec<([u8; 32], u64)>) -> String {
    records.sort_unstable();

    let mut hasher = Sha256::new();
    hasher.update(b"kiln.base-model-content.v1\0");
    hasher.update((records.len() as u64).to_le_bytes());
    for (digest, byte_len) in records {
        hasher.update(byte_len.to_le_bytes());
        hasher.update(digest);
    }
    let digest = hasher.finalize();
    format!(
        "sha256:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

/// Complete Qwen3.5-4B language model weights.
///
/// Note: lm_head is tied to embed_tokens (shared weight matrix),
/// so we don't store it separately.
#[derive(Debug)]
pub struct ModelWeights {
    /// Content revision of the exact safetensors shard bytes memory-mapped by
    /// the loader, independent of checkpoint path, shard order, and mtimes.
    /// Test-constructed weights that did not pass through the file loader have
    /// no authoritative source revision.
    pub source_content_sha256: Option<String>,
    /// Retains the exact loader mappings and their open files until the caller
    /// has verified the source again after GPU upload.
    pub(crate) source_content_guard: Option<SourceContentGuard>,
    pub embedding: EmbeddingWeights,
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm. [hidden_size]
    pub final_norm: WeightTensor,
    /// Optional native MTP head (Qwen3.5-4B ships one, other variants may not).
    /// Populated when `num_nextn_predict_layers > 0` in the model config AND the
    /// `mtp.*` tensors are present in the checkpoint. Consumed by
    /// `KILN_SPEC_METHOD=mtp` at serve time.
    pub mtp: Option<MtpWeights>,
    /// Optional deferred native-MTP source. When present, the checkpoint ships
    /// `mtp.*` tensors but the caller elected not to materialize them during
    /// initial model load.
    pub deferred_mtp: Option<DeferredMtpSource>,
}

impl ModelWeights {
    /// Verify that the loader's exact source shards still match the revision
    /// recorded before safetensors parsing.
    ///
    /// Call this immediately after GPU upload. Verification reads the retained
    /// open files in a fixed-size buffer; it does not rediscover paths or trust
    /// mtimes, and truncation is reported without dereferencing an invalidated
    /// mmap region.
    pub fn verify_source_content_unchanged(&self) -> Result<()> {
        let guard = self
            .source_content_guard
            .as_ref()
            .context("model weights have no loader-owned source content guard")?;
        if self.source_content_sha256.as_deref() != Some(guard.initial_sha256()) {
            bail!(
                "model source revision does not match the loader-owned revision: expected {}, observed {}",
                guard.initial_sha256(),
                self.source_content_sha256.as_deref().unwrap_or("missing")
            );
        }
        guard.verify_unchanged()
    }

    /// Total size of all loaded weights in bytes.
    pub fn total_bytes(&self) -> usize {
        let mut total = self.embedding.embed_tokens.size_bytes();
        total += self.final_norm.size_bytes();
        if let Some(mtp) = &self.mtp {
            total += mtp.total_bytes();
        }
        for layer in &self.layers {
            total += layer.input_layernorm.size_bytes();
            total += layer.post_attention_layernorm.size_bytes();
            total += layer.mlp.gate_proj.size_bytes();
            total += layer.mlp.up_proj.size_bytes();
            total += layer.mlp.down_proj.size_bytes();
            match &layer.attention {
                AttentionWeights::Full(attn) => {
                    total += attn.q_proj.size_bytes();
                    total += attn.k_proj.size_bytes();
                    total += attn.v_proj.size_bytes();
                    total += attn.o_proj.size_bytes();
                    total += attn.q_norm.size_bytes();
                    total += attn.k_norm.size_bytes();
                }
                AttentionWeights::Linear(attn) => {
                    total += attn.in_proj_qkv.size_bytes();
                    total += attn.in_proj_z.size_bytes();
                    total += attn.out_proj.size_bytes();
                    total += attn.in_proj_a.size_bytes();
                    total += attn.in_proj_b.size_bytes();
                    total += attn.conv1d.size_bytes();
                    total += attn.norm.size_bytes();
                    total += attn.a_log.size_bytes();
                    total += attn.dt_bias.size_bytes();
                }
            }
        }
        total
    }

    /// Total number of parameters.
    pub fn total_params(&self) -> usize {
        let mut total = self.embedding.embed_tokens.numel();
        total += self.final_norm.numel();
        if let Some(mtp) = &self.mtp {
            total += mtp.total_params();
        }
        for layer in &self.layers {
            total += layer.input_layernorm.numel();
            total += layer.post_attention_layernorm.numel();
            total += layer.mlp.gate_proj.numel();
            total += layer.mlp.up_proj.numel();
            total += layer.mlp.down_proj.numel();
            match &layer.attention {
                AttentionWeights::Full(attn) => {
                    total += attn.q_proj.numel();
                    total += attn.k_proj.numel();
                    total += attn.v_proj.numel();
                    total += attn.o_proj.numel();
                    total += attn.q_norm.numel();
                    total += attn.k_norm.numel();
                }
                AttentionWeights::Linear(attn) => {
                    total += attn.in_proj_qkv.numel();
                    total += attn.in_proj_z.numel();
                    total += attn.out_proj.numel();
                    total += attn.in_proj_a.numel();
                    total += attn.in_proj_b.numel();
                    total += attn.conv1d.numel();
                    total += attn.norm.numel();
                    total += attn.a_log.numel();
                    total += attn.dt_bias.numel();
                }
            }
        }
        total
    }
}
