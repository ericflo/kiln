pub mod adapter_merge;
pub mod backend;
pub mod cancel;
pub mod cuda_graph;
// (#1082 Wave F2) `cuda_train` deleted — the hand-rolled candle-autograd
// CUDA training engine (`CudaTrainTensor`/`CudaBackwardOp`/`cuda_backward`)
// is gone; the kt tape is the sole gradient producer.
pub mod decode_buffers;
pub mod engine;
pub mod forward;
pub mod fp8;
mod full_attention_policy;
pub mod generate;
mod kt_api_policy;
pub mod kv_cache;
pub mod loader;
pub mod lora;
pub mod lora_loader;
pub mod marlin_proj;
pub mod metal_graph;
mod mtp_runtime;
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
    TrainingAccelerationProfileLogMessage, TrainingAccelerationProfilePolicy,
    TrainingOptimizerKind, TrainingOptimizerRequest, TrainingOptimizerRounding,
    TrainingOptimizerSupport, TrainingOptimizerSupportError,
};
pub use backend::residency::{
    ReplayStability, ResidentOwnership, ResidentRegistry, ResidentResource, ResidentResourceFamily,
    ResidentResourceLayout, ResidentResourceState, resident_backend_for_runtime,
    resident_ownership_for_backend,
};
pub use backend::{
    AttentionBackend, BackendIdentity, BackendRuntime, ConvBackend, ExternalYieldBackend,
    FallbackPolicy, GdnBackend, GdnRecurrentStateResidencyStats, LinearBackend, OptimizerBackend,
    PagedKvBackend, ReplayBackend, ResidencyBackend, SamplingBackend, StartupBackend,
    TrainingLossBackend, TrainingPrecisionPolicy,
};
// (#1082 candle removal) `backend::for_device` (candle-typed shim) was deleted
// with the candle-parity opt-in feature; production uses `for_device_kt`.
#[cfg(feature = "rocm")]
pub use backend::rocm::{RocmKernelPolicy, install_rocm_kernel_policy};
pub use backend::{DecodeWeightPrewarmCancelled, DecodeWeightPrewarmPolicy};
pub use cancel::CancelHandle;
pub use engine::Engine;
pub use forward::{LinearAttentionState, StreamingPrefillExecutionPolicy, StreamingPrefillMode};
pub use full_attention_policy::{
    DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB, MAX_FULL_ATTENTION_SCORE_BUDGET_MIB,
    MIN_FULL_ATTENTION_SCORE_BUDGET_MIB, install_full_attention_score_budget_mib,
    validate_full_attention_score_budget_mib,
};
pub use generate::{
    BackendHealthHandle, BackendHealthSnapshot, BatchedStateCacheStats, DecodeBatcher,
    DecodeBatcherConfig, DecodeBatcherStats, ExternalYieldSyncStats, FinishReason,
    GenerationOutput, InferenceMemoryRuntime, ModelRunner, ModelRunnerRuntimeOptions,
    MtpGenerationOutput, PagedBatchedDecodeState, PagedBatchedPrefillProgress,
    PagedBatchedPrefillStart, PagedBatchedPrefillState, PagedPrefixNextToken,
    PagedPrefixRegistration, PagedPrefixReuse, PrefixCachedGenerationOutput, StreamDone,
    StreamEvent, StreamToken, ThreadedStreamingOutput,
};
pub use kt_api_policy::{KtApiMode, install_kt_api_mode};
pub use kv_cache::KvCache;
pub use loader::{
    LoadModelOptions, load_model, load_model_with_options, load_model_with_options_and_snapshot_dir,
};
pub use lora_loader::LoraWeights;
pub use paged_kv_cache_kt::{KvPoolIdentity, PagedKvCacheKt};
pub use rocm_graph::{
    RocmGraphExecutionMode, RocmGraphExecutionPolicy, RocmGraphFallbackStats,
    RocmGraphLiveTelemetry, RocmGraphPhase, RocmGraphPhaseStats, RocmGraphStats,
    RocmGraphStatsUnavailable, RocmGraphTelemetryHandle,
};
pub use speculative::SpeculativeConfig;
pub use weights::{ModelSnapshotCleanup, ModelWeights};

#[cfg(feature = "vulkan")]
pub use kiln_vulkan_kernel::kernels::VULKAN_KERNEL_POLICY_SCHEMA_ID;
#[cfg(not(feature = "vulkan"))]
pub const VULKAN_KERNEL_POLICY_SCHEMA_ID: &str = "kiln.vulkan-kernel-policy.v3";

#[cfg(feature = "vulkan")]
pub use kiln_vulkan_kernel::{
    VULKAN_DEVICE_POLICY_SCHEMA_ID, VulkanDevicePolicy, install_vulkan_device_policy,
};
#[cfg(not(feature = "vulkan"))]
pub const VULKAN_DEVICE_POLICY_SCHEMA_ID: &str = "kiln.vulkan-device-policy.v1";

#[cfg(feature = "vulkan")]
pub use kiln_vulkan_kernel::buffer::VulkanBufferAllocationStats;
#[cfg(feature = "vulkan")]
pub use kiln_vulkan_kernel::buffer_pool::{
    BufferPoolCacheMiss as VulkanBufferPoolCacheMiss,
    BufferPoolCacheMissRoute as VulkanBufferPoolCacheMissRoute,
    BufferPoolStats as VulkanBufferPoolStats,
};

#[cfg(not(feature = "vulkan"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VulkanBufferAllocationStats {
    pub live_device_local_buffers: u64,
    pub live_device_local_bytes: u64,
    pub live_host_visible_buffers: u64,
    pub live_host_visible_bytes: u64,
    pub peak_live_bytes: u64,
    pub device_local_allocations: u64,
    pub device_local_allocated_bytes: u64,
    pub device_local_frees: u64,
    pub device_local_freed_bytes: u64,
    pub host_visible_allocations: u64,
    pub host_visible_allocated_bytes: u64,
    pub host_visible_frees: u64,
    pub host_visible_freed_bytes: u64,
}

#[cfg(not(feature = "vulkan"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VulkanBufferPoolCacheMissRoute {
    #[default]
    None,
    DeviceLocal,
    HostVisible,
}

#[cfg(not(feature = "vulkan"))]
impl VulkanBufferPoolCacheMissRoute {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::DeviceLocal => "device_local",
            Self::HostVisible => "host_visible",
        }
    }
}

#[cfg(not(feature = "vulkan"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VulkanBufferPoolCacheMiss {
    pub sequence: u64,
    pub route: VulkanBufferPoolCacheMissRoute,
    pub requested_bytes: u64,
    pub bucket_bytes: u64,
    pub caller_file: &'static str,
    pub caller_line: u32,
}

#[cfg(not(feature = "vulkan"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VulkanBufferPoolStats {
    pub max_retained_bytes: u64,
    pub bucket_count: usize,
    pub buffer_count: usize,
    pub total_bytes: u64,
    pub free_buffer_count: usize,
    pub free_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub device_local_cache_misses: u64,
    pub host_visible_cache_misses: u64,
    pub last_cache_miss: VulkanBufferPoolCacheMiss,
    pub eviction_count: u64,
    pub evicted_bytes: u64,
    pub uncached_allocation_count: u64,
    pub uncached_allocated_bytes: u64,
}

#[cfg(not(feature = "vulkan"))]
impl VulkanBufferPoolStats {
    pub fn borrowed_buffer_count(self) -> usize {
        self.buffer_count.saturating_sub(self.free_buffer_count)
    }

    pub fn borrowed_bytes(self) -> u64 {
        self.total_bytes.saturating_sub(self.free_bytes)
    }
}

pub fn vulkan_buffer_allocation_stats() -> Option<VulkanBufferAllocationStats> {
    #[cfg(feature = "vulkan")]
    {
        Some(kiln_vulkan_kernel::buffer::allocation_stats())
    }
    #[cfg(not(feature = "vulkan"))]
    {
        None
    }
}

pub fn vulkan_buffer_pool_stats() -> Option<VulkanBufferPoolStats> {
    #[cfg(feature = "vulkan")]
    {
        Some(kiln_vulkan_kernel::buffer_pool::pool_stats())
    }
    #[cfg(not(feature = "vulkan"))]
    {
        None
    }
}

pub fn configure_vulkan_buffer_pool(max_retained_bytes: u64) -> u64 {
    #[cfg(feature = "vulkan")]
    {
        kiln_vulkan_kernel::buffer_pool::pool_configure_max_retained_bytes(max_retained_bytes)
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = max_retained_bytes;
        0
    }
}

pub fn trim_vulkan_buffer_pool(target_bytes: u64) -> u64 {
    #[cfg(feature = "vulkan")]
    {
        kiln_vulkan_kernel::buffer_pool::pool_trim_free(target_bytes)
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = target_bytes;
        0
    }
}
