pub mod adapter_merge;
pub mod backend;
pub mod c1_attr;
pub mod cancel;
pub mod cuda_graph;
// (#1082 Wave F2) `cuda_train` deleted — the hand-rolled candle-autograd
// CUDA training engine (`CudaTrainTensor`/`CudaBackwardOp`/`cuda_backward`)
// is gone; the kt tape is the sole gradient producer.
pub mod decode_buffers;
pub mod engine;
pub mod forward;
pub mod fp8;
pub mod generate;
pub mod kv_cache;
pub mod loader;
pub mod lora;
pub mod lora_loader;
pub mod marlin_proj;
pub mod metal_graph;
pub mod mtp_debug;
pub mod packed_weight_registry;
pub mod rocm_graph;
// (#1082 all-hardware) `paged_kv_cache_kt` compiles on every backend now: the
// struct, metadata accessors, and `new`/`new_with_fp8` constructors are
// available everywhere (CPU-resident pools on the Vulkan/CPU build), while the
// CUDA-kernel write/read methods stay `#[cfg(feature = "cuda")]` inside the
// module. This is required because `forward.rs`/`generate.rs`/`cuda_graph.rs`/
// `speculative.rs`/`vk_decode_resident.rs` thread `&PagedKvCacheKt` through
// signatures that are NOT cuda-gated.
pub mod paged_kv_cache_kt;
pub mod quantized;
pub mod qwen35_shapes;
pub mod rocm_w8_proj;
pub mod sampling;
pub mod speculative;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
pub mod tape_forward;
mod transposed_weight_cache;
#[cfg(feature = "vulkan")]
pub mod vk_bwd_adapter;
#[cfg(feature = "vulkan")]
pub mod vk_decode_resident;
pub mod weights;

pub use backend::{
    AttentionBackend, BackendIdentity, BackendRuntime, ConvBackend, FallbackPolicy,
    GdnBackend, LinearBackend, OptimizerBackend, PagedKvBackend, ReplayBackend,
    ResidencyBackend, SamplingBackend, TrainingLossBackend, TrainingPrecisionPolicy,
};
pub use backend::capability::{
    AttentionCapabilities, AttentionRequest, AttentionRequestKind, BackendCapabilityQueries,
    BackendCapabilitySnapshot, BackendCapabilities, BackendFallbackCapabilities,
    BackendTrainingCapabilities, DecodeCapabilities, GdnCapabilities, LinearRequest,
    LinearRequestKind, MatmulAccumulation, MatmulBatchPolicy, MatmulBlasRequest,
    MatmulCapabilities, MatmulEpilogue, MatmulOperandLayout, MatmulRequest,
    MatmulRequestProjectionError, ReplayCapabilities, ReplayRequest, ReplayRequestKind,
    StorageCapabilities, Support,
};
pub use backend::residency::{
    ReplayStability, ResidentOwnership, ResidentRegistry, ResidentResource,
    ResidentResourceFamily, ResidentResourceLayout, ResidentResourceState,
    resident_ownership_for_backend,
};
// (#1082 candle removal) `backend::for_device` (candle-typed shim) was deleted
// with the candle-parity opt-in feature; production uses `for_device_kt`.
pub use cancel::CancelHandle;
pub use engine::Engine;
pub use forward::LinearAttentionState;
pub use generate::{
    DecodeBatcher, DecodeBatcherConfig, DecodeBatcherStats, FinishReason, GenerationOutput,
    ModelRunner, MtpGenerationOutput, PagedBatchedDecodeState, PagedPrefixNextToken,
    PagedPrefixRegistration, PagedPrefixReuse, PrefixCachedGenerationOutput, StreamDone,
    StreamEvent, StreamToken,
};
pub use kv_cache::KvCache;
pub use loader::{LoadModelOptions, load_model, load_model_with_options};
pub use lora_loader::LoraWeights;
pub use paged_kv_cache_kt::PagedKvCacheKt;
pub use speculative::SpeculativeConfig;
pub use weights::ModelWeights;
