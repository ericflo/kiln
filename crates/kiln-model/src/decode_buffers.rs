//! Pre-allocated typed buffers for the Qwen3.5 decode executor.
//!
//! Phase A keeps allocation and shape validation explicit. These buffers are
//! allocated at runner construction/warmup and then reused by decode iterations,
//! so later kernels can consume stable device addresses under CUDA graph capture
//! instead of asking Candle to allocate, reshape, or materialize layout views on
//! the production decode path.

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor};
use std::fmt;
use std::marker::PhantomData;

use crate::qwen35_shapes as shapes;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeElementType {
    Bf16,
    Fp8E4M3,
    Fp32,
}

impl DecodeElementType {
    pub fn candle_dtype(self) -> DType {
        match self {
            Self::Bf16 => DType::BF16,
            Self::Fp8E4M3 => DType::U8,
            Self::Fp32 => DType::F32,
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp8E4M3 => 1,
            Self::Fp32 => 4,
        }
    }
}

pub enum Bf16 {}
pub enum Fp8E4M3 {}
pub enum Fp32 {}

pub trait DecodeDType {
    const ELEMENT_TYPE: DecodeElementType;
}

impl DecodeDType for Bf16 {
    const ELEMENT_TYPE: DecodeElementType = DecodeElementType::Bf16;
}

impl DecodeDType for Fp8E4M3 {
    const ELEMENT_TYPE: DecodeElementType = DecodeElementType::Fp8E4M3;
}

impl DecodeDType for Fp32 {
    const ELEMENT_TYPE: DecodeElementType = DecodeElementType::Fp32;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeBufferKind {
    Hidden,
    Q,
    K,
    V,
    GdnState,
    KvCachePages,
    Logits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferShape {
    dims: &'static [usize],
}

impl BufferShape {
    pub const fn new(dims: &'static [usize]) -> Self {
        Self { dims }
    }

    pub fn dims(self) -> &'static [usize] {
        self.dims
    }

    pub fn elements(self) -> usize {
        self.dims.iter().product()
    }
}

pub struct DecodeBuffer<T: DecodeDType> {
    kind: DecodeBufferKind,
    tensor: Tensor,
    dims: Vec<usize>,
    _dtype: PhantomData<T>,
}

impl<T: DecodeDType> fmt::Debug for DecodeBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecodeBuffer")
            .field("kind", &self.kind)
            .field("dims", &self.dims)
            .field("dtype", &T::ELEMENT_TYPE)
            .finish()
    }
}

impl<T: DecodeDType> DecodeBuffer<T> {
    pub fn allocate(kind: DecodeBufferKind, dims: impl Into<Vec<usize>>, device: &Device) -> Result<Self> {
        let dims = dims.into();
        ensure!(!dims.is_empty(), "decode buffer {kind:?} must have at least one dimension");
        ensure!(dims.iter().all(|&dim| dim > 0), "decode buffer {kind:?} dimensions must be non-zero: {dims:?}");
        let tensor = Tensor::zeros(dims.as_slice(), T::ELEMENT_TYPE.candle_dtype(), device)
            .with_context(|| format!("allocate decode buffer {kind:?} {dims:?}"))?;
        ensure!(tensor.is_contiguous(), "decode buffer {kind:?} allocation must be contiguous");
        Ok(Self { kind, tensor, dims, _dtype: PhantomData })
    }

    pub fn kind(&self) -> DecodeBufferKind {
        self.kind
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn element_type(&self) -> DecodeElementType {
        T::ELEMENT_TYPE
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn tensor_mut(&mut self) -> &mut Tensor {
        &mut self.tensor
    }

    pub fn byte_len(&self) -> usize {
        self.dims.iter().product::<usize>() * T::ELEMENT_TYPE.bytes()
    }

    #[cfg(feature = "cuda")]
    pub fn with_bf16_device_ptr<R>(&self, f: impl FnOnce(*const core::ffi::c_void) -> R) -> Result<R> {
        use half::bf16;

        ensure!(T::ELEMENT_TYPE == DecodeElementType::Bf16, "decode buffer {:?} is not BF16", self.kind);
        ensure!(self.tensor.is_contiguous(), "decode buffer {:?} must be contiguous for raw pointer access", self.kind);
        let (storage, layout) = self.tensor.storage_and_layout();
        let cuda = match &*storage {
            candle_core::Storage::Cuda(cuda) => cuda,
            _ => bail!("decode buffer {:?} must be on CUDA for raw pointer access", self.kind),
        };
        let stream = cuda.device().cuda_stream();
        let slice = cuda.as_cuda_slice::<bf16>()?.slice(layout.start_offset()..);
        unsafe {
            let (ptr, _guard) = slice.device_ptr(&stream);
            Ok(f(ptr as *const core::ffi::c_void))
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecodeBufferConfig {
    pub max_batch: usize,
    pub max_context: usize,
    pub kv_pages: usize,
    pub page_size: usize,
    pub kv_dtype: DecodeElementType,
}

impl DecodeBufferConfig {
    pub fn graph_bucket(max_batch: usize, max_context: usize, kv_pages: usize, page_size: usize, kv_dtype: DecodeElementType) -> Result<Self> {
        ensure!(max_batch > 0, "decode buffer max_batch must be non-zero");
        ensure!(max_context > 0, "decode buffer max_context must be non-zero");
        ensure!(kv_pages > 0, "decode buffer kv_pages must be non-zero");
        ensure!(page_size > 0, "decode buffer page_size must be non-zero");
        ensure!(matches!(kv_dtype, DecodeElementType::Bf16 | DecodeElementType::Fp8E4M3), "KV cache dtype must be BF16 or FP8");
        Ok(Self { max_batch, max_context, kv_pages, page_size, kv_dtype })
    }

    pub fn hidden_dims(&self) -> [usize; 2] {
        [self.max_batch, shapes::HIDDEN]
    }

    pub fn full_q_dims(&self) -> [usize; 3] {
        [self.max_batch, shapes::NUM_HEADS, shapes::HEAD_DIM]
    }

    pub fn full_kv_dims(&self) -> [usize; 3] {
        [self.max_batch, shapes::NUM_KV_HEADS, shapes::HEAD_DIM]
    }

    pub fn gdn_state_dims(&self) -> [usize; 5] {
        [
            self.max_batch,
            shapes::NUM_GDN_LAYERS,
            shapes::GDN_NUM_VALUE_HEADS,
            shapes::GDN_KEY_HEAD_DIM,
            shapes::GDN_VALUE_HEAD_DIM,
        ]
    }

    pub fn kv_page_dims(&self) -> [usize; 5] {
        [
            self.kv_pages,
            shapes::NUM_FULL_ATTN_LAYERS,
            2,
            self.page_size,
            shapes::FULL_KV_WIDTH,
        ]
    }

    pub fn logits_dims(&self) -> [usize; 2] {
        [self.max_batch, shapes::VOCAB]
    }
}

pub enum KvCachePageBuffer {
    Bf16(DecodeBuffer<Bf16>),
    Fp8(DecodeBuffer<Fp8E4M3>),
}

impl KvCachePageBuffer {
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::Bf16(buffer) => buffer.dims(),
            Self::Fp8(buffer) => buffer.dims(),
        }
    }

    pub fn element_type(&self) -> DecodeElementType {
        match self {
            Self::Bf16(buffer) => buffer.element_type(),
            Self::Fp8(buffer) => buffer.element_type(),
        }
    }
}

pub struct DecodeBuffers {
    pub config: DecodeBufferConfig,
    pub hidden: DecodeBuffer<Bf16>,
    pub q: DecodeBuffer<Bf16>,
    pub k: DecodeBuffer<Bf16>,
    pub v: DecodeBuffer<Bf16>,
    pub gdn_state: DecodeBuffer<Bf16>,
    pub kv_pages: KvCachePageBuffer,
    pub logits: DecodeBuffer<Fp32>,
}

impl DecodeBuffers {
    pub fn allocate(config: DecodeBufferConfig, device: &Device) -> Result<Self> {
        let hidden = DecodeBuffer::allocate(DecodeBufferKind::Hidden, config.hidden_dims(), device)?;
        let q = DecodeBuffer::allocate(DecodeBufferKind::Q, config.full_q_dims(), device)?;
        let k = DecodeBuffer::allocate(DecodeBufferKind::K, config.full_kv_dims(), device)?;
        let v = DecodeBuffer::allocate(DecodeBufferKind::V, config.full_kv_dims(), device)?;
        let gdn_state = DecodeBuffer::allocate(DecodeBufferKind::GdnState, config.gdn_state_dims(), device)?;
        let kv_pages = match config.kv_dtype {
            DecodeElementType::Bf16 => KvCachePageBuffer::Bf16(DecodeBuffer::allocate(DecodeBufferKind::KvCachePages, config.kv_page_dims(), device)?),
            DecodeElementType::Fp8E4M3 => KvCachePageBuffer::Fp8(DecodeBuffer::allocate(DecodeBufferKind::KvCachePages, config.kv_page_dims(), device)?),
            DecodeElementType::Fp32 => bail!("decode KV pages cannot use FP32"),
        };
        let logits = DecodeBuffer::allocate(DecodeBufferKind::Logits, config.logits_dims(), device)?;
        Ok(Self { config, hidden, q, k, v, gdn_state, kv_pages, logits })
    }

    pub fn ensure_batch_fits(&self, batch: usize) -> Result<()> {
        ensure!(batch > 0, "decode batch must be non-zero");
        ensure!(batch <= self.config.max_batch, "decode batch {batch} exceeds buffer max_batch {}", self.config.max_batch);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_bucket_shapes_are_qwen35_baked() {
        let cfg = DecodeBufferConfig::graph_bucket(8, 1024, 256, 16, DecodeElementType::Bf16).unwrap();
        assert_eq!(cfg.hidden_dims(), [8, shapes::HIDDEN]);
        assert_eq!(cfg.full_q_dims(), [8, shapes::NUM_HEADS, shapes::HEAD_DIM]);
        assert_eq!(cfg.full_kv_dims(), [8, shapes::NUM_KV_HEADS, shapes::HEAD_DIM]);
        assert_eq!(cfg.gdn_state_dims(), [8, shapes::NUM_GDN_LAYERS, shapes::GDN_NUM_VALUE_HEADS, shapes::GDN_KEY_HEAD_DIM, shapes::GDN_VALUE_HEAD_DIM]);
        assert_eq!(cfg.kv_page_dims(), [256, shapes::NUM_FULL_ATTN_LAYERS, 2, 16, shapes::FULL_KV_WIDTH]);
        assert_eq!(cfg.logits_dims(), [8, shapes::VOCAB]);
    }

    #[test]
    fn graph_bucket_rejects_invalid_capacity() {
        assert!(DecodeBufferConfig::graph_bucket(0, 1024, 256, 16, DecodeElementType::Bf16).is_err());
        assert!(DecodeBufferConfig::graph_bucket(8, 1024, 256, 16, DecodeElementType::Fp32).is_err());
    }

    #[test]
    fn decode_buffer_metadata_tracks_bytes() {
        let device = Device::Cpu;
        let buffer = DecodeBuffer::<Fp32>::allocate(DecodeBufferKind::Logits, [2, 4], &device).unwrap();
        assert_eq!(buffer.dims(), &[2, 4]);
        assert_eq!(buffer.byte_len(), 32);
        assert_eq!(buffer.element_type(), DecodeElementType::Fp32);
    }
}
