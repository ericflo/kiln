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
pub mod stream_text;
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

pub use backend::capability::{
    AttentionCapabilities, AttentionRequest, AttentionRequestKind, BackendCapabilities,
    BackendCapabilityQueries, BackendCapabilitySnapshot, BackendFallbackCapabilities,
    BackendTrainingCapabilities, DecodeBatcherPolicy, DecodeCapabilities, GdnCapabilities,
    GpuAllocatorMemoryProbe, GpuAllocatorMemoryProbePolicy, GpuMemoryBudgetPolicy,
    GpuMemoryDetectionPolicy, GpuMemoryReclaimPolicy, GpuMemoryReclaimer,
    InferenceRecurrentStatePolicy, KvCacheAutoBlockPolicy, KvCacheFp8Policy,
    KvCacheMemoryTierBlockCap, LinearRequest, LinearRequestKind, MatmulAccumulation,
    MatmulBatchPolicy, MatmulBlasRequest, MatmulCapabilities, MatmulEpilogue, MatmulOperandLayout,
    MatmulRequest, MatmulRequestProjectionError, ReplayAuthority, ReplayCapabilities,
    ReplayGraphCrateRole, ReplayNativePrimitive, ReplayProductionAuthority, ReplayRequest,
    ReplayRequestKind, ServerTrainingDispatchPolicy, ServerTrainingNativeRoute,
    SpeculativeDecodePolicy, StartupCapabilities, StorageCapabilities,
    StreamingPrefillAutoDispatch, StreamingPrefillBackendPolicy, Support,
    TrainingAccelerationEnvFlagPolicy, TrainingAccelerationProfileLogMessage,
    TrainingAccelerationProfilePolicy, TrainingOptimizerKind, TrainingOptimizerRequest,
    TrainingOptimizerRounding, TrainingOptimizerSupport, TrainingOptimizerSupportError,
};
pub use backend::residency::{
    ReplayStability, ResidentOwnership, ResidentRegistry, ResidentResource, ResidentResourceFamily,
    ResidentResourceLayout, ResidentResourceState, resident_backend_for_runtime,
    resident_ownership_for_backend,
};
pub use backend::{
    AttentionBackend, BackendIdentity, BackendRuntime, ConvBackend, ExternalYieldBackend,
    FallbackPolicy, GdnBackend, LinearBackend, OptimizerBackend, PagedKvBackend, ReplayBackend,
    ResidencyBackend, SamplingBackend, StartupBackend, TrainingLossBackend,
    TrainingPrecisionPolicy,
};
// (#1082 candle removal) `backend::for_device` (candle-typed shim) was deleted
// with the candle-parity opt-in feature; production uses `for_device_kt`.
pub use cancel::CancelHandle;
pub use engine::Engine;
pub use forward::{LinearAttentionState, StreamingPrefillExecutionPolicy, StreamingPrefillMode};
pub use generate::{
    BackendHealthHandle, BackendHealthSnapshot, DecodeBatcher, DecodeBatcherConfig,
    DecodeBatcherStats, ExternalYieldSyncStats, FinishReason, GenerationOutput,
    InferenceMemoryRuntime, ModelRunner, ModelRunnerRuntimeOptions, MtpGenerationOutput,
    PagedBatchedDecodeState, PagedBatchedPrefillProgress, PagedBatchedPrefillStart,
    PagedBatchedPrefillState, PagedPrefixNextToken, PagedPrefixRegistration, PagedPrefixReuse,
    PrefixCachedGenerationOutput, StreamDone, StreamEvent, StreamToken, ThreadedStreamingOutput,
};
pub use kv_cache::KvCache;
pub use loader::{
    LoadModelOptions, load_model, load_model_with_options, load_model_with_options_and_snapshot_dir,
};
pub use lora_loader::LoraWeights;
pub use paged_kv_cache_kt::{KvPoolIdentity, PagedKvCacheKt};
pub use rocm_graph::{RocmGraphFallbackStats, RocmGraphStats};
pub use speculative::SpeculativeConfig;
pub use weights::{ModelSnapshotCleanup, ModelWeights};
