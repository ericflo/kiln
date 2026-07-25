//! CUDA backend: FlashAttention-2 and Gated DeltaNet fused kernels.
//!
//! Wraps the vendored `kiln-flash-attn` and `kiln-gdn-kernel` crates.
//! `Ok(None)` responses route the caller to the portable candle path.

use anyhow::{Context, Result};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    AttentionBackend, BackendIdentity, BackendMatmulLayout, BackendRuntime, ConvBackend,
    ExternalYieldBackend, GdnBackend, LinearBackend, OptimizerBackend, PagedKvBackend,
    ReplayBackend, ResidencyBackend, SamplingBackend, StartupBackend, TrainingCapabilities,
    TrainingLossBackend, TrainingPrecisionPolicy, matmul_request_support_rank,
    matmul_support_from_native, requested_matmul_layout,
};
use crate::cuda_policy::{CudaKernelPolicy, current_cuda_kernel_policy};
use crate::lora_loader::{LoraProjectionWeights, compute_lora_delta};

static CUDA_SGD_DISPATCH_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static CUDA_ADAMW_DISPATCH_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static CUDA_MUON_DISPATCH_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static CUDA_LINEAR_PREFILL_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static CUDA_LINEAR_PREFILL_OFFSET_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static CUDA_FLASH_ATTN_TRACKED_DECLINES: AtomicU64 = AtomicU64::new(0);
static CUDA_GDN_FULL_CHUNK_FORWARD_MULTIBLOCK_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static CUDA_GDN_FULL_CHUNK_FORWARD_SINGLE_SUCCESSES: AtomicU64 = AtomicU64::new(0);

pub fn optimizer_dispatch_success_counts() -> (u64, u64) {
    (
        CUDA_SGD_DISPATCH_SUCCESSES.load(Ordering::Relaxed),
        CUDA_ADAMW_DISPATCH_SUCCESSES.load(Ordering::Relaxed),
    )
}

pub fn reset_optimizer_dispatch_success_counts() {
    CUDA_SGD_DISPATCH_SUCCESSES.store(0, Ordering::Relaxed);
    CUDA_ADAMW_DISPATCH_SUCCESSES.store(0, Ordering::Relaxed);
}

pub fn linear_prefill_success_counts() -> (u64, u64) {
    (
        CUDA_LINEAR_PREFILL_SUCCESSES.load(Ordering::Relaxed),
        CUDA_LINEAR_PREFILL_OFFSET_SUCCESSES.load(Ordering::Relaxed),
    )
}

pub fn reset_linear_prefill_success_counts() {
    CUDA_LINEAR_PREFILL_SUCCESSES.store(0, Ordering::Relaxed);
    CUDA_LINEAR_PREFILL_OFFSET_SUCCESSES.store(0, Ordering::Relaxed);
}

const CUDA_GDN_PREFILL_AB_IN_PROJ_MAX_TOKENS: usize = 128;

pub fn flash_attn_tracked_decline_count() -> u64 {
    CUDA_FLASH_ATTN_TRACKED_DECLINES.load(Ordering::Relaxed)
}

pub fn reset_flash_attn_tracked_decline_count() {
    CUDA_FLASH_ATTN_TRACKED_DECLINES.store(0, Ordering::Relaxed);
}

/// `(multiblock_path_successes, single_block_path_successes)` for
/// `gdn_full_chunk_forward`. Used by tests and bench tooling to confirm which
/// kernel actually ran under a given typed policy.
pub fn gdn_full_chunk_forward_dispatch_counts() -> (u64, u64) {
    (
        CUDA_GDN_FULL_CHUNK_FORWARD_MULTIBLOCK_SUCCESSES.load(Ordering::Relaxed),
        CUDA_GDN_FULL_CHUNK_FORWARD_SINGLE_SUCCESSES.load(Ordering::Relaxed),
    )
}

pub fn reset_gdn_full_chunk_forward_dispatch_counts() {
    CUDA_GDN_FULL_CHUNK_FORWARD_MULTIBLOCK_SUCCESSES.store(0, Ordering::Relaxed);
    CUDA_GDN_FULL_CHUNK_FORWARD_SINGLE_SUCCESSES.store(0, Ordering::Relaxed);
}

const CUDA_RESIDENT_TENSOR_IDS_POISONED: &str = "CUDA resident TensorId registry mutex poisoned";

fn cuda_optimizer_args_ready_for_kt(
    registry: &super::cuda_rocm_common::ResidentTensorIdRegistry,
    tensors: &[&kiln_tensor::Tensor],
) -> bool {
    super::cuda_rocm_common::optimizer_args_ready_for_kt(
        registry,
        tensors,
        CUDA_RESIDENT_TENSOR_IDS_POISONED,
        |device| matches!(device, kiln_tensor::Device::Cuda(_)),
    )
}

fn cuda_tensors_on_device(tensors: &[&kiln_tensor::Tensor]) -> bool {
    super::cuda_rocm_common::tensors_on_backend_device(tensors, |device| {
        matches!(device, kiln_tensor::Device::Cuda(_))
    })
}

#[derive(Debug)]
pub struct CudaBackend {
    /// The kt CUDA device this backend was constructed for. (#1082 DoD-100
    /// step 4: the formerly-cached candle `device` field was dropped — it had
    /// zero reads; `new` now takes a `kiln_tensor::Device` directly.)
    device_kt: kiln_tensor::Device,
    resident_tensor_ids: super::cuda_rocm_common::ResidentTensorIdRegistry,
    full_attn_qkv_in_proj_enabled: bool,
    gdn_ab_in_proj_enabled: bool,
    gdn_prefill_ab_in_proj_enabled: bool,
    gdn_prefill_gates_enabled: bool,
    /// Cached from immutable startup policy before any dispatch.
    gdn_enabled: bool,
    /// Fused gates authority from the same immutable policy.
    gdn_gates_enabled: bool,
    /// Typed route for fused GDN gated RMSNorm in decode and prefill.
    gdn_gated_rms_norm_enabled: bool,
    /// Experimental fused native-MTP decode GDN gates + recurrent update.
    /// Opt-in only until output parity is proven.
    gdn_decode_fused_enabled: bool,
    /// CUDA fused decode supports native GQA Q/K heads; this avoids expanding
    /// Q/K to value_heads before the fused recurrent decode kernel.
    gdn_decode_unexpanded_qk_enabled: bool,
    /// Fuses GDN decode Q/K L2-normalization into the gates+recurrent kernel,
    /// avoiding the separate tiny qk_norm launch in the single-token path.
    gdn_decode_qk_norm_recurrent_enabled: bool,
    /// Fuses GDN decode Q/K L2-normalization, gates, recurrent update, and
    /// gated RMSNorm into one single-token CUDA launch.
    gdn_decode_qk_norm_recurrent_rmsnorm_enabled: bool,
    /// Typed route for fused causal convolution. When off, `forward.rs` uses
    /// the portable reference chain.
    fused_conv1d_enabled: bool,
    // Phase 7 (#1082): the cuda_use_kt_api_conv1d gate was removed once
    // the kt-typed surface (causal_conv1d_{update,prefill}_kt +
    // supports{,_prefill}_kt) became the only path. The complete startup
    // profile now owns fallback for the convolution family.
    // Phase 7 (#1082): the cuda_use_kt_api_gdn gate was removed once
    // all 10 GDN dispatch wires (forward_substitution, recurrent_step,
    // chunk_prep, chunk_scan, full_chunk_forward[_multiblock],
    // gates, gated_rms_norm, plus the 4 decode_* wires:
    // gates_recurrent, qk_norm_gates_recurrent,
    // qk_norm_gates_recurrent_rmsnorm) became kt-only. The startup profile
    // selects the complete GDN family and its portable fallbacks. The kt-typed
    // path is bit-exact with the previous kt-API code (same FFI
    // symbol). All 11 GDN dispatch wires (including the formerly
    // candle-typed single-block `gdn_full_chunk_forward` fall-through
    // inside the multiblock dispatcher) are now kt-only after the
    // `gdn_full_chunk_forward_kt` single-block wire landed.
    // Phase 7 (#1082): all 4 flash-attn dispatch sites in this
    // backend are now kt-only after `aab07fa7` landed the
    // `flash_attn_paged_decode_dyn_seqlen_kt_with_graph_outputs`
    // sibling. The `cuda_use_kt_api_flash_attn` gate is gone. The
    // `_with_graph_outputs` site dispatches both branches through
    // kt: `Some((out, lse))` borrows the caller's candle tensors
    // into kt and writes through them via the new with_graph_outputs
    // entry; `None` calls the existing internally-allocating
    // `flash_attn_paged_decode_dyn_seqlen_kt`.
    /// Forward-only CUDA LoRA delta/add for decode. Training declines because
    /// tracked LoRA tensors need autograd.
    lora_decode_add_enabled: bool,
    /// Multi-block dv-tiled `gdn_full_chunk_forward`. The native default keeps
    /// it on because the
    /// single-block kernel only launches `B*H = 32` blocks for Qwen3.5-4B at
    /// batch=1, leaving ~58% of a 76-SM RTX 4090 Laptop idle. The multi-block
    /// path is bit-exact with the legacy kernel (same per-output-cell FMA
    /// chain, same bf16 rounding). Portable fallback declines it.
    gdn_full_chunk_forward_multiblock_enabled: bool,
}

impl CudaBackend {
    pub fn new(device: kiln_tensor::Device) -> Self {
        Self::new_with_kernel_policy(device, current_cuda_kernel_policy())
    }

    fn new_with_kernel_policy(device: kiln_tensor::Device, policy: CudaKernelPolicy) -> Self {
        debug_assert!(
            matches!(device, kiln_tensor::Device::Cuda(_)),
            "CudaBackend created on non-CUDA device"
        );
        let gdn_enabled = policy.gdn;
        let gdn_gates_enabled = gdn_enabled && policy.gdn_gates;
        let gdn_gated_rms_norm_enabled = gdn_enabled && policy.gdn_gated_rms_norm;
        let fused_conv1d_enabled = policy.fused_conv1d;
        // #1082: the dedicated per-operation KT disable gate was
        // removed once the kt-typed conv1d surface became the only
        // path in `causal_conv1d_{update,prefill}`. Profile selection now owns
        // the complete native-versus-portable choice.
        // #1082: the dedicated per-operation KT disable gate was
        // removed once the kt-typed GDN surface became the only path
        // across all 10 dispatch wires. Profile selection now owns the GDN
        // family and its portable reference paths.
        // #1082: flipped default ON. The kt path is bit-exact by
        // construction — all 4 wired flash_attn dispatch sites bottom
        // out in the same FFI symbols as the candle shim, with only
        // the Rust shell types changing. The `with_graph_outputs`
        // site retains its `graph_outputs.is_none()` guard so the
        // caller-owned-output path keeps using the candle wrapper.
        // The per-operation KT disable gate was removed alongside the
        // 3 sites where the kt-typed path is the only path. The
        // 4th site checks `graph_outputs.is_none()` directly.
        let gdn_decode_fused_enabled =
            gdn_gates_enabled && gdn_gated_rms_norm_enabled && policy.gdn_decode_fused;
        let gdn_decode_unexpanded_qk_enabled =
            gdn_decode_fused_enabled && policy.gdn_decode_unexpanded_qk;
        let gdn_decode_qk_norm_recurrent_enabled =
            gdn_decode_unexpanded_qk_enabled && policy.gdn_decode_qk_norm_recurrent;
        let gdn_decode_qk_norm_recurrent_rmsnorm_enabled =
            gdn_decode_qk_norm_recurrent_enabled && policy.gdn_decode_qk_norm_recurrent_rmsnorm;
        let gdn_full_chunk_forward_multiblock_enabled =
            gdn_enabled && policy.gdn_full_chunk_forward_multiblock;
        let device_kt = device;
        Self {
            device_kt,
            resident_tensor_ids: super::cuda_rocm_common::new_resident_tensor_id_registry(),
            full_attn_qkv_in_proj_enabled: policy.full_attn_qkv_in_proj,
            gdn_ab_in_proj_enabled: gdn_enabled && policy.gdn_ab_in_proj,
            gdn_prefill_ab_in_proj_enabled: gdn_enabled && policy.gdn_prefill_ab_in_proj,
            gdn_prefill_gates_enabled: gdn_gates_enabled && policy.gdn_prefill_gates,
            gdn_enabled,
            gdn_gates_enabled,
            gdn_gated_rms_norm_enabled,
            gdn_decode_fused_enabled,
            gdn_decode_unexpanded_qk_enabled,
            gdn_decode_qk_norm_recurrent_enabled,
            gdn_decode_qk_norm_recurrent_rmsnorm_enabled,
            fused_conv1d_enabled,
            lora_decode_add_enabled: policy.lora_decode_add,
            gdn_full_chunk_forward_multiblock_enabled,
        }
    }

    pub fn training_capabilities_static() -> TrainingCapabilities {
        TrainingCapabilities {
            projection_training: "backend-routed kt cublasLt matmul (tape-recorded) with offset chunk hook",
            flce_loss: "FLCE analytic backward on CUDA tensors; no full logits by default",
            tape_forward_backward_route: super::TrainingTapeRoute::KtTapeAuthoritative,
            sft_flce_loss_route: super::SftFlceLossRoute::KtTapeFlce,
            grpo_loss_route: super::GrpoLossRoute::KtComposite,
            grpo_kl_auxiliary_route: super::GrpoKlAuxiliaryRoute::CudaRocmDeviceFastPath,
            opd_loss_route: super::OpdLossRoute::KtTapePhaseB,
            opd_phase_b_backward_route: super::OpdPhaseBBackwardRoute::CudaRocmFusedUnitGrad,
            final_rmsnorm_backward_route: super::FinalRmsNormBackwardRoute::CudaRocmFusedTail,
            rmsnorm_training: "CUDA kt-tape rmsnorm behind 47 GiB autograd VRAM gate",
            resident_activation: "kt TensorId lifecycle registry; kt CUDA tensors are canonical",
            lora_delta_training: "kt tape-recorded LoRA delta; fused lora_decode_add declines tape-tracked tensors",
            sgd_step: "CUDA in-place optimizer kernel for resident contiguous F32/BF16 tensors",
            adamw_step: "CUDA in-place optimizer kernel for resident contiguous F32/BF16 tensors",
            native_training: "not implemented",
        }
    }

    fn support_predicates(&self) -> super::cuda_rocm_common::CudaRocmSupportPredicates {
        super::cuda_rocm_common::CudaRocmSupportPredicates {
            gdn_enabled: self.gdn_enabled,
            gdn_gates_enabled: self.gdn_gates_enabled,
            gdn_gated_rms_norm_enabled: self.gdn_gated_rms_norm_enabled,
            gdn_decode_unexpanded_qk_enabled: self.gdn_decode_unexpanded_qk_enabled,
            gdn_decode_qk_norm_recurrent_enabled: self.gdn_decode_qk_norm_recurrent_enabled,
            fused_conv1d_enabled: self.fused_conv1d_enabled,
        }
    }
}

impl BackendIdentity for CudaBackend {
    fn runtime_name(&self) -> &'static str {
        "cuda"
    }

    fn runtime_device(&self) -> kiln_tensor::Device {
        self.device_kt
    }

    fn runtime_as_any(&self) -> &dyn std::any::Any {
        &()
    }
}

impl StartupBackend for CudaBackend {}

impl ExternalYieldBackend for CudaBackend {
    fn runtime_synchronize_external_yield(&self) -> Result<()> {
        let kiln_tensor::Device::Cuda(device_index) = self.device_kt else {
            anyhow::bail!("CUDA external-yield synchronization requires a CUDA device");
        };
        let context = kiln_tensor::primary_cuda_context(device_index)
            .context("acquire CUDA context for external-yield synchronization")?;
        context
            .bind_to_thread()
            .context("bind CUDA context for external-yield synchronization")?;
        kiln_tensor::cuda_synchronize_context_for(
            device_index,
            &context,
            kiln_tensor::CudaSyncReason::ExternalYield,
        )
        .context("synchronize CUDA context before external yield")
    }
}

#[allow(clippy::too_many_arguments)]
impl AttentionBackend for CudaBackend {
    fn runtime_supports_flash_attn_prefill(&self) -> bool {
        self.support_predicates().supports_flash_attn_prefill()
    }

    fn runtime_supports_flash_attn_paged_decode(&self) -> bool {
        self.support_predicates().supports_flash_attn_paged_decode()
    }

    /// CUDA has no impl for the strict `flash_attn_paged_decode_contiguous_batch`
    /// kernel (the bs>1 head-major uniform-`start_pos` path), so the trait
    /// default `Ok(None)` always declines. Returning `false` here lets the
    /// `try_strict` probe in `gqa_attention_paged_decode_contiguous_batch`
    /// skip the `start_slots = Tensor::from_slice(...)` allocation that
    /// would otherwise emit a captured `cudaMemcpyHtoDAsync` to a recycled
    /// VA under CUDA graph capture (suspect 6 in
    /// `bench-results/cuda-graph-bs2-secondary-audit.md`, #1082).
    fn runtime_supports_strict_paged_decode_contiguous_batch(&self) -> bool {
        self.support_predicates()
            .supports_strict_paged_decode_contiguous_batch()
    }

    fn runtime_flash_attn_prefill(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // The vendored CUDA kernel hard-errors on non-BF16. Decline here so
        // the caller falls back to the portable path instead of bubbling a
        // hard error up for non-BF16 test configs.
        if q.track_op() || k.track_op() || v.track_op() {
            CUDA_FLASH_ATTN_TRACKED_DECLINES.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        if q.dtype() != kiln_tensor::DType::BF16 {
            return Ok(None);
        }
        // The vendored FA-2 kernel only supports head_dim 128/256 and
        // hard-errors otherwise. Decline (mirroring the BF16 decline above)
        // so non-{128,256} configs — e.g. the head_dim=16 tiny test model on
        // the detached CP-4 tape-authoritative path, which clears the
        // track_op + BF16 gates above — fall back to the portable SDPA path
        // instead of bubbling a hard error. (#1082)
        if !matches!(q.dims().last(), Some(&128) | Some(&256)) {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed surface is now the only path. Args
        // are already kt (#1082 DoD-101/102), so the candle↔kt bridges
        // are gone — the kernel runs directly on the caller's kt
        // tensors. candle wrapper discards softmax_lse, kt path does
        // the same here.
        kiln_nvtx::range!(c"kiln/flash_attn_kt");
        let q_c = q.contiguous().context("flash_attn kt: q contiguous")?;
        let k_c = k.contiguous().context("flash_attn kt: k contiguous")?;
        let v_c = v.contiguous().context("flash_attn kt: v contiguous")?;
        let (out_kt, _lse_kt) =
            kiln_flash_attn::flash_attn_fwd_kt(&q_c, &k_c, &v_c, softmax_scale, causal)
                .map_err(|e| anyhow::anyhow!("flash_attn kt: flash_attn_fwd_kt: {e}"))?;
        Ok(Some(out_kt))
    }

    fn runtime_flash_attn_paged_decode(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        total_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if q.track_op()
            || k_pool.track_op()
            || v_pool.track_op()
            || block_table.track_op()
            || q.dtype() != kiln_tensor::DType::BF16
        {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-only. Args are already kt (#1082
        // DoD-101/102), so no candle↔kt bridge — the kernel runs
        // directly on the caller's kt tensors.
        kiln_nvtx::range!(c"kiln/flash_attn_paged_decode_kt");
        let out_kt = kiln_flash_attn::flash_attn_paged_decode_kt(
            q,
            k_pool,
            v_pool,
            block_table,
            total_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
        .map_err(|e| anyhow::anyhow!("flash_attn_paged_decode kt: {e}"))?;
        Ok(Some(out_kt))
    }

    fn runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        seqused_k: &kiln_tensor::Tensor,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if q.track_op()
            || k_pool.track_op()
            || v_pool.track_op()
            || block_table.track_op()
            || seqused_k.track_op()
            || q.dtype() != kiln_tensor::DType::BF16
        {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-only. Args are already kt (#1082
        // DoD-101/102), so no candle↔kt bridge. This entry always
        // passed `graph_outputs = None`; the caller-owned-output
        // variant lives in `_with_graph_outputs` below.
        kiln_nvtx::range!(c"kiln/flash_attn_paged_decode_dyn_seqlen_kt");
        let out_kt = kiln_flash_attn::flash_attn_paged_decode_dyn_seqlen_kt(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
        .map_err(|e| anyhow::anyhow!("flash_attn_paged_decode_dyn_seqlen kt: {e}"))?;
        Ok(Some(out_kt))
    }
}

#[allow(clippy::too_many_arguments)]
impl OptimizerBackend for CudaBackend {
    fn runtime_dispatch_sgd_step(
        &self,
        param: &kiln_tensor::Tensor,
        grad: &kiln_tensor::Tensor,
        lr: f32,
    ) -> Result<bool> {
        if !cuda_optimizer_args_ready_for_kt(&self.resident_tensor_ids, &[param, grad]) {
            return Ok(false);
        }
        // #1082: args are already kt (BackendRuntime trait flipped to kt),
        // so there is no candle<->kt borrow - the kernel runs directly on the
        // caller's kt tensors. Bit-exact with the prior candle shim (same
        // FFI symbol).
        kiln_nvtx::range!(c"kiln/sgd_step_kt");
        match param.dtype() {
            kiln_tensor::DType::F32 => kiln_rmsnorm_kernel::sgd_step_f32_kt(param, grad, lr)
                .map_err(|e| anyhow::anyhow!("sgd_step kt: sgd_step_f32_kt: {e}"))?,
            kiln_tensor::DType::BF16 => kiln_rmsnorm_kernel::sgd_step_bf16_kt(param, grad, lr)
                .map_err(|e| anyhow::anyhow!("sgd_step kt: sgd_step_bf16_kt: {e}"))?,
            other => anyhow::bail!("sgd_step kt: unsupported dtype {other:?}"),
        }
        CUDA_SGD_DISPATCH_SUCCESSES.fetch_add(1, Ordering::Relaxed);
        static FIRST_CUDA_SGD_LOGGED: OnceLock<()> = OnceLock::new();
        FIRST_CUDA_SGD_LOGGED.get_or_init(|| {
            tracing::info!(
                param_shape = ?param.dims(),
                grad_shape = ?grad.dims(),
                dtype = ?param.dtype(),
                lr,
                "CudaBackend::dispatch_sgd_step first call"
            );
        });
        Ok(true)
    }

    fn runtime_dispatch_adamw_step(
        &self,
        param: &kiln_tensor::Tensor,
        grad: &kiln_tensor::Tensor,
        first_moment: &kiln_tensor::Tensor,
        second_moment: &kiln_tensor::Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> Result<bool> {
        if !cuda_optimizer_args_ready_for_kt(
            &self.resident_tensor_ids,
            &[param, grad, first_moment, second_moment],
        ) {
            return Ok(false);
        }
        // #1082: args are already kt (BackendRuntime trait flipped to kt),
        // so there is no candle<->kt borrow - the kernel runs directly on the
        // caller's kt tensors. Bit-exact with the prior candle shim (same
        // FFI symbol).
        if step == 0 {
            anyhow::bail!("adamw_step kt: step must be >= 1");
        }
        kiln_nvtx::range!(c"kiln/adamw_step_kt");
        let bias_correction1 = (1.0f32 - beta1.powi(step as i32)).max(1e-20);
        let bias_correction2 = (1.0f32 - beta2.powi(step as i32)).max(1e-20);
        match param.dtype() {
            kiln_tensor::DType::F32 => kiln_rmsnorm_kernel::adamw_step_f32_kt(
                param,
                grad,
                first_moment,
                second_moment,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                bias_correction1,
                bias_correction2,
            )
            .map_err(|e| anyhow::anyhow!("adamw_step kt: adamw_step_f32_kt: {e}"))?,
            kiln_tensor::DType::BF16 => kiln_rmsnorm_kernel::adamw_step_bf16_kt(
                param,
                grad,
                first_moment,
                second_moment,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                bias_correction1,
                bias_correction2,
            )
            .map_err(|e| anyhow::anyhow!("adamw_step kt: adamw_step_bf16_kt: {e}"))?,
            other => anyhow::bail!("adamw_step kt: unsupported dtype {other:?}"),
        }
        CUDA_ADAMW_DISPATCH_SUCCESSES.fetch_add(1, Ordering::Relaxed);
        static FIRST_CUDA_ADAMW_LOGGED: OnceLock<()> = OnceLock::new();
        FIRST_CUDA_ADAMW_LOGGED.get_or_init(|| {
            tracing::info!(
                param_shape = ?param.dims(),
                grad_shape = ?grad.dims(),
                first_moment_shape = ?first_moment.dims(),
                second_moment_shape = ?second_moment.dims(),
                dtype = ?param.dtype(),
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
                "CudaBackend::dispatch_adamw_step first call"
            );
        });
        Ok(true)
    }

    fn runtime_dispatch_muon_step(
        &self,
        param: &kiln_tensor::Tensor,
        grad: &kiln_tensor::Tensor,
        momentum: &kiln_tensor::Tensor,
        lr: f32,
        momentum_coef: f32,
        nesterov: bool,
        ns_iters: u32,
        weight_decay: f32,
    ) -> Result<bool> {
        if !cuda_optimizer_args_ready_for_kt(&self.resident_tensor_ids, &[param, grad, momentum]) {
            return Ok(false);
        }
        // #1082: args are already kt (BackendRuntime trait flipped to kt),
        // so there is no candle<->kt borrow - the kernel runs directly on the
        // caller's kt tensors. Updates param + momentum in place in one launch.
        kiln_nvtx::range!(c"kiln/muon_step_kt");
        match param.dtype() {
            kiln_tensor::DType::F32 => kiln_rmsnorm_kernel::muon_step_f32_kt(
                param,
                grad,
                momentum,
                lr,
                momentum_coef,
                nesterov,
                ns_iters,
                weight_decay,
            )
            .map_err(|e| anyhow::anyhow!("muon_step kt: muon_step_f32_kt: {e}"))?,
            kiln_tensor::DType::BF16 => kiln_rmsnorm_kernel::muon_step_bf16_kt(
                param,
                grad,
                momentum,
                lr,
                momentum_coef,
                nesterov,
                ns_iters,
                weight_decay,
            )
            .map_err(|e| anyhow::anyhow!("muon_step kt: muon_step_bf16_kt: {e}"))?,
            other => anyhow::bail!("muon_step kt: unsupported dtype {other:?}"),
        }
        CUDA_MUON_DISPATCH_SUCCESSES.fetch_add(1, Ordering::Relaxed);
        static FIRST_CUDA_MUON_LOGGED: OnceLock<()> = OnceLock::new();
        FIRST_CUDA_MUON_LOGGED.get_or_init(|| {
            tracing::info!(
                param_shape = ?param.dims(),
                grad_shape = ?grad.dims(),
                momentum_shape = ?momentum.dims(),
                dtype = ?param.dtype(),
                lr,
                momentum_coef,
                nesterov,
                ns_iters,
                weight_decay,
                "CudaBackend::dispatch_muon_step first call"
            );
        });
        Ok(true)
    }
}

#[allow(clippy::too_many_arguments)]
impl GdnBackend for CudaBackend {
    fn runtime_supports_gdn_forward_substitution(&self) -> bool {
        self.support_predicates()
            .supports_gdn_forward_substitution()
    }

    fn runtime_supports_gdn_recurrent_step(&self) -> bool {
        self.support_predicates().supports_gdn_recurrent_step()
    }

    fn runtime_supports_gdn_chunk_prep(&self) -> bool {
        self.support_predicates().supports_gdn_chunk_prep()
    }

    fn runtime_supports_gdn_chunk_scan(&self) -> bool {
        self.support_predicates().supports_gdn_chunk_scan()
    }

    fn runtime_supports_gdn_full_chunk_forward(&self) -> bool {
        self.support_predicates().supports_gdn_full_chunk_forward()
    }

    fn runtime_supports_gdn_decode_gates_recurrent_unexpanded_qk(&self) -> bool {
        self.support_predicates()
            .supports_gdn_decode_gates_recurrent_unexpanded_qk()
    }

    fn runtime_supports_gdn_decode_qk_norm_gates_recurrent(&self) -> bool {
        self.support_predicates()
            .supports_gdn_decode_qk_norm_gates_recurrent()
    }

    fn runtime_gdn_forward_substitution(
        &self,
        a_strict: &kiln_tensor::Tensor,
        v_prime: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // Phase 7 (#1082): kt-typed surface is now the only path. With
        // the BackendRuntime decode methods flipped to kt (#1082
        // DoD-101/102) the args are already kt, so the candle↔kt
        // bridges are gone — the kernel runs directly on the caller's
        // kt tensors (zero candle roundtrip).
        kiln_nvtx::range!(c"kiln/gdn_forward_substitution_kt");
        let out_kt = match a_strict.dtype() {
            kiln_tensor::DType::BF16 => {
                kiln_gdn_kernel::gdn_forward_substitution_kt(a_strict, v_prime, beta)
                    .map_err(|e| anyhow::anyhow!("kt gdn_forward_substitution: {e}"))?
            }
            kiln_tensor::DType::F32 => {
                kiln_gdn_kernel::gdn_forward_substitution_f32_kt(a_strict, v_prime, beta)
                    .map_err(|e| anyhow::anyhow!("kt gdn_forward_substitution_f32: {e}"))?
            }
            _ => return Ok(None),
        };
        Ok(Some(out_kt))
    }

    fn runtime_gdn_solve_tri_transpose(
        &self,
        a_strict: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        dw: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if a_strict.dtype() != kiln_tensor::DType::F32 {
            return Ok(None);
        }
        kiln_nvtx::range!(c"kiln/gdn_solve_tri_transpose_f32_kt");
        let out_kt = kiln_gdn_kernel::gdn_solve_tri_transpose_f32_kt(a_strict, beta, dw)
            .map_err(|e| anyhow::anyhow!("kt gdn_solve_tri_transpose_f32: {e}"))?;
        Ok(Some(out_kt))
    }

    fn runtime_gdn_recurrent_step(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if q.dtype() != kiln_tensor::DType::BF16 {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed surface is now the only path. Args
        // are already kt (#1082 DoD-101/102), so no candle↔kt bridge —
        // the kernel's in-place mutation of `state` surfaces through
        // the caller's `&mut kt::Tensor`.
        kiln_nvtx::range!(c"kiln/gdn_recurrent_forward_kt");
        let out_kt = kiln_gdn_kernel::gdn_recurrent_forward_kt(q, k, v, beta, g, state)
            .map_err(|e| anyhow::anyhow!("kt gdn_recurrent_forward: {e}"))?;
        Ok(Some(out_kt))
    }

    fn runtime_gdn_chunk_prep(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        // Phase 7 (#1082): kt-typed surface is now the only path. Args
        // are already kt (#1082 DoD-101/102), so no candle↔kt bridge —
        // the predicate + `gdn_chunk_prep_kt` 6-tuple kernel run
        // directly on the caller's kt tensors and the kt outputs are
        // returned without a copy.
        kiln_nvtx::range!(c"kiln/gdn_chunk_prep_kt");
        if !kiln_gdn_kernel::gdn_chunk_prep_supports_kt(g, v, kkt, qkt, ks_entry, q_s) {
            return Ok(None);
        }
        let (o0, o1, o2, o3, o4, o5) =
            kiln_gdn_kernel::gdn_chunk_prep_kt(g, v, kkt, qkt, ks_entry, q_s)
                .map_err(|e| anyhow::anyhow!("kt gdn_chunk_prep: {e}"))?;
        Ok(Some((o0, o1, o2, o3, o4, o5)))
    }

    fn runtime_gdn_chunk_scan(
        &self,
        a_strict: &kiln_tensor::Tensor,
        b_mask: &kiln_tensor::Tensor,
        v_prime: &kiln_tensor::Tensor,
        q_s_scaled: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        decay_last_col: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        // Phase 7 (#1082): kt-typed surface is now the only path. Args
        // are already kt (#1082 DoD-101/102), so no candle↔kt bridge.
        kiln_nvtx::range!(c"kiln/gdn_chunk_scan_kt");
        if !kiln_gdn_kernel::gdn_chunk_scan_supports_kt(
            a_strict,
            b_mask,
            v_prime,
            q_s_scaled,
            beta,
            decay_last_col,
        ) {
            return Ok(None);
        }
        let (o0, o1) = kiln_gdn_kernel::gdn_chunk_scan_kt(
            a_strict,
            b_mask,
            v_prime,
            q_s_scaled,
            beta,
            decay_last_col,
        )
        .map_err(|e| anyhow::anyhow!("kt gdn_chunk_scan: {e}"))?;
        Ok(Some((o0, o1)))
    }

    fn runtime_gdn_full_chunk_forward(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        k_t: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        let dv_tile = kiln_gdn_kernel::GDN_FULL_CHUNK_FORWARD_MULTIBLOCK_DV_TILE;
        // Phase 7 (#1082): kt-typed surface is now the only path
        // (single-block + multiblock). Args are already kt (#1082
        // DoD-101/102), so no candle↔kt bridge — both predicates and
        // kernels run directly on the caller's kt tensors. State
        // mutation surfaces through the caller's `&mut kt::Tensor`.
        if !kiln_gdn_kernel::gdn_full_chunk_forward_supports_kt(
            g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state,
        ) {
            return Ok(None);
        }
        if self.gdn_full_chunk_forward_multiblock_enabled
            && kiln_gdn_kernel::gdn_full_chunk_forward_multiblock_supports_kt(
                g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state, dv_tile,
            )
        {
            kiln_nvtx::range!(c"kiln/gdn_full_chunk_forward_multiblock_kt");
            let out_kt = kiln_gdn_kernel::gdn_full_chunk_forward_multiblock_kt(
                g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state, dv_tile,
            )
            .map_err(|e| anyhow::anyhow!("kt gdn_full_chunk_forward_multiblock: {e}"))?;
            CUDA_GDN_FULL_CHUNK_FORWARD_MULTIBLOCK_SUCCESSES.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(out_kt));
        }
        // Single-block fall-through. The `gdn_full_chunk_forward_kt`
        // wire bottoms out in the same FFI symbol; `state` mutation
        // surfaces through the caller's `&mut kt::Tensor`.
        kiln_nvtx::range!(c"kiln/gdn_full_chunk_forward_kt");
        let out_kt = kiln_gdn_kernel::gdn_full_chunk_forward_kt(
            g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state,
        )
        .map_err(|e| anyhow::anyhow!("kt gdn_full_chunk_forward: {e}"))?;
        CUDA_GDN_FULL_CHUNK_FORWARD_SINGLE_SUCCESSES.fetch_add(1, Ordering::Relaxed);
        Ok(Some(out_kt))
    }

    #[allow(clippy::too_many_arguments)]
    fn runtime_gdn_decode_gates_recurrent(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !self.gdn_decode_fused_enabled {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed bf16 surface is now the only path.
        // Args are already kt (#1082 DoD-101/102), so no candle↔kt
        // bridge — the kernel runs directly on the caller's kt
        // tensors. Non-bf16 envelopes return Ok(None) so the caller's
        // split-decode fallback engages. Production decode for
        // Qwen3.5-4B uses bf16 for q/k/v/a/b/a_log/dt_bias/state/z and
        // f32 for the rmsnorm weight; the bf16_kt variant is the
        // matching production hot path.
        if !(q.dtype() == kiln_tensor::DType::BF16
            && k.dtype() == kiln_tensor::DType::BF16
            && v.dtype() == kiln_tensor::DType::BF16
            && a.dtype() == kiln_tensor::DType::BF16
            && b.dtype() == kiln_tensor::DType::BF16
            && a_log.dtype() == kiln_tensor::DType::BF16
            && dt_bias.dtype() == kiln_tensor::DType::BF16
            && state.dtype() == kiln_tensor::DType::BF16
            && z.dtype() == kiln_tensor::DType::BF16
            && weight.dtype() == kiln_tensor::DType::F32)
        {
            tracing::debug!(
                q_shape = ?q.shape(), q_dtype = ?q.dtype(),
                k_shape = ?k.shape(), k_dtype = ?k.dtype(),
                v_shape = ?v.shape(), v_dtype = ?v.dtype(),
                a_shape = ?a.shape(), a_dtype = ?a.dtype(),
                b_shape = ?b.shape(), b_dtype = ?b.dtype(),
                a_log_shape = ?a_log.shape(), a_log_dtype = ?a_log.dtype(),
                dt_bias_shape = ?dt_bias.shape(), dt_bias_dtype = ?dt_bias.dtype(),
                state_shape = ?state.shape(), state_dtype = ?state.dtype(), state_contiguous = state.is_contiguous(),
                z_shape = ?z.shape(), z_dtype = ?z.dtype(),
                weight_shape = ?weight.shape(), weight_dtype = ?weight.dtype(),
                "CUDA gdn_decode_gates_recurrent declined (non-bf16 envelope); will retry with cast"
            );
            // Phase 5 fix (#1082): same dtype-tolerance pattern as
            // `gdn_decode_qk_norm_gates_recurrent` above. Cast small
            // 1-D weight tensors (a_log, dt_bias) to BF16 if they
            // arrived as F32 from the safetensors loader; the
            // group-norm weight stays F32 (kernel contract).
            let a_log_bf16 = if a_log.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    a_log
                        .to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_gates: cast a_log -> bf16")?,
                )
            };
            let dt_bias_bf16 = if dt_bias.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    dt_bias
                        .to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_gates: cast dt_bias -> bf16")?,
                )
            };
            let weight_f32 = if weight.dtype() == kiln_tensor::DType::F32 {
                None
            } else {
                Some(
                    weight
                        .to_dtype(kiln_tensor::DType::F32)
                        .with_context(|| "gdn_decode_gates: cast weight -> f32")?,
                )
            };
            // Cast heavy tensors too. q/k/v often arrive F32 from
            // the conv1d kernel epilogue (kernel returns F32 by
            // design — see forward.rs:14897). The kt path is BF16-only
            // — cast at the boundary so the kt surface still works for
            // the production hot path. Cast cost is per-layer
            // hidden-dim elements (small at B=1). `state` is NOT cast
            // — the kernel mutates it in place and the caller's tensor
            // would not see the writes through a fresh allocation.
            if state.dtype() != kiln_tensor::DType::BF16 {
                return Ok(None);
            }
            let q_bf16 = if q.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    q.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_gates: cast q -> bf16")?,
                )
            };
            let k_bf16 = if k.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    k.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_gates: cast k -> bf16")?,
                )
            };
            let v_bf16 = if v.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    v.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_gates: cast v -> bf16")?,
                )
            };
            let a_bf16 = if a.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    a.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_gates: cast a -> bf16")?,
                )
            };
            let b_bf16 = if b.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    b.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_gates: cast b -> bf16")?,
                )
            };
            let z_bf16 = if z.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    z.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_gates: cast z -> bf16")?,
                )
            };
            return self.runtime_gdn_decode_gates_recurrent(
                q_bf16.as_ref().unwrap_or(q),
                k_bf16.as_ref().unwrap_or(k),
                v_bf16.as_ref().unwrap_or(v),
                a_bf16.as_ref().unwrap_or(a),
                b_bf16.as_ref().unwrap_or(b),
                a_log_bf16.as_ref().unwrap_or(a_log),
                dt_bias_bf16.as_ref().unwrap_or(dt_bias),
                state,
                z_bf16.as_ref().unwrap_or(z),
                weight_f32.as_ref().unwrap_or(weight),
                eps,
            );
        }
        kiln_nvtx::range!(c"kiln/gdn_decode_gates_recurrent_bf16_kt");
        // Run supports_kt on the ORIGINAL 4D tensors — the kt
        // predicate's shape contract is 4D `[B, 1, q_heads, dk]`
        // per kt_api.rs:2199-2203. Then squeeze for the kernel call
        // which expects 3D `[B, heads, dim]`. The kernel/predicate
        // shape mismatch was a latent bug surfaced by the Phase 5
        // sanitizer sweep (the 3D-pre-predicate ordering made
        // supports_kt always return false → caller declined).
        //
        // #1082 bench regression (2026-05-26): same non-contig
        // hazard as gdn_decode_qk_norm_gates_recurrent_rmsnorm below
        // — in the batched concurrent decode path, `a`/`b` arrive as
        // `ab.narrow(2, .., nv)` views on the fused in-proj output
        // and need an explicit `.contiguous()` before the kt borrow.
        // a_log/dt_bias/state/z/weight are also materialized here
        // for symmetry with gdn_gates and to future-proof against
        // upstream batched-shape changes (no-op on already-contig).
        let a_c = a
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_gates a contiguous failed")?;
        let b_c = b
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_gates b contiguous failed")?;
        let alog_c = a_log
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_gates a_log contiguous failed")?;
        let dtb_c = dt_bias
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_gates dt_bias contiguous failed")?;
        let state_c = state
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_gates state contiguous failed")?;
        let z_c = z
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_gates z contiguous failed")?;
        let weight_c = weight
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_gates weight contiguous failed")?;
        if !kiln_gdn_kernel::gdn_decode_gates_recurrent_supports_kt(
            q, k, v, &a_c, &b_c, &alog_c, &dtb_c, &state_c, &z_c, &weight_c,
        ) {
            return Ok(None);
        }
        // Squeeze for the kernel call (3D-expecting). Metadata-only
        // reshape on contiguous inputs; no copy.
        let q_3d = q
            .squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_gates q squeeze(1) failed")?;
        let k_3d = k
            .squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_gates k squeeze(1) failed")?;
        let v_3d = v
            .squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_gates v squeeze(1) failed")?;
        let out_kt = kiln_gdn_kernel::gdn_decode_gates_recurrent_bf16_kt(
            &q_3d, &k_3d, &v_3d, &a_c, &b_c, &alog_c, &dtb_c, &state_c, &z_c, &weight_c, eps as f32,
        )
        .map_err(|e| anyhow::anyhow!("kt gdn_decode_gates_recurrent: {e}"))?;
        // kt_api allocates a 3D `[B, value_heads, dv]` output (see
        // crates/kiln-gdn-kernel/src/kt_api.rs:568) but the
        // BackendRuntime trait + production caller expect 4D
        // `[B, 1, value_heads, dv]` (see attn_out shape contract at
        // forward.rs:15595). Unsqueeze the seq_len axis back at
        // position 1; metadata-only reshape, no copy. (#1082)
        let out = out_kt
            .unsqueeze(1)
            .with_context(|| "kt-adapter: gdn_decode_gates out 3D->4D unsqueeze failed")?;
        Ok(Some(out))
    }

    #[allow(clippy::too_many_arguments)]
    fn runtime_gdn_decode_qk_norm_gates_recurrent(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        q_scale: f64,
        qk_eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !self.gdn_decode_qk_norm_recurrent_enabled {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed bf16 surface is now the only path.
        // Args are already kt (#1082 DoD-101/102), so no candle↔kt
        // bridge. Non-bf16 envelopes return Ok(None) so the caller's
        // split qk_norm fallback engages. Production decode for
        // Qwen3.5-4B uses bf16 for all 8 input tensors; the bf16_kt
        // variant is the matching production hot path.
        if !(q.dtype() == kiln_tensor::DType::BF16
            && k.dtype() == kiln_tensor::DType::BF16
            && v.dtype() == kiln_tensor::DType::BF16
            && a.dtype() == kiln_tensor::DType::BF16
            && b.dtype() == kiln_tensor::DType::BF16
            && a_log.dtype() == kiln_tensor::DType::BF16
            && dt_bias.dtype() == kiln_tensor::DType::BF16
            && state.dtype() == kiln_tensor::DType::BF16)
        {
            tracing::debug!(
                q_shape = ?q.shape(), q_dtype = ?q.dtype(),
                k_shape = ?k.shape(), k_dtype = ?k.dtype(),
                v_shape = ?v.shape(), v_dtype = ?v.dtype(),
                a_shape = ?a.shape(), a_dtype = ?a.dtype(),
                b_shape = ?b.shape(), b_dtype = ?b.dtype(),
                a_log_shape = ?a_log.shape(), a_log_dtype = ?a_log.dtype(),
                dt_bias_shape = ?dt_bias.shape(), dt_bias_dtype = ?dt_bias.dtype(),
                state_shape = ?state.shape(), state_dtype = ?state.dtype(), state_contiguous = state.is_contiguous(),
                "CUDA gdn_decode_qk_norm_gates_recurrent declined (non-bf16 envelope); will retry with cast"
            );
            // Phase 5 fix (#1082): Qwen3.5-4B safetensors store
            // `A_log` and `dt_bias` in F32 by default (loader keeps
            // these "non-linear weights" as-is at loader.rs:897). The
            // kt path requires every input in BF16, so cast the small
            // 1-D weight tensors here and recurse. The cast is cheap
            // (num_heads elements) and one-shot per call — the cost
            // is dominated by the kernel launch.
            let a_log_bf16 = if a_log.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    a_log
                        .to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_qk_norm: cast a_log -> bf16")?,
                )
            };
            let dt_bias_bf16 = if dt_bias.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    dt_bias
                        .to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_qk_norm: cast dt_bias -> bf16")?,
                )
            };
            // Same heavy-tensor cast as gdn_decode_gates_recurrent
            // above — required because conv1d kernel emits F32
            // q/k/v.
            if state.dtype() != kiln_tensor::DType::BF16 {
                return Ok(None);
            }
            let q_bf16 = if q.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    q.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_qk_norm: cast q -> bf16")?,
                )
            };
            let k_bf16 = if k.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    k.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_qk_norm: cast k -> bf16")?,
                )
            };
            let v_bf16 = if v.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    v.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_qk_norm: cast v -> bf16")?,
                )
            };
            let a_bf16 = if a.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    a.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_qk_norm: cast a -> bf16")?,
                )
            };
            let b_bf16 = if b.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    b.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_qk_norm: cast b -> bf16")?,
                )
            };
            return self.runtime_gdn_decode_qk_norm_gates_recurrent(
                q_bf16.as_ref().unwrap_or(q),
                k_bf16.as_ref().unwrap_or(k),
                v_bf16.as_ref().unwrap_or(v),
                a_bf16.as_ref().unwrap_or(a),
                b_bf16.as_ref().unwrap_or(b),
                a_log_bf16.as_ref().unwrap_or(a_log),
                dt_bias_bf16.as_ref().unwrap_or(dt_bias),
                state,
                q_scale,
                qk_eps,
            );
        }
        kiln_nvtx::range!(c"kiln/gdn_decode_qk_norm_gates_recurrent_bf16_kt");
        // supports_kt on 4D, kernel on 3D — see gdn_decode_gates_recurrent
        // above for the same shape-contract split.
        //
        // #1082 bench regression (2026-05-26): same non-contig hazard
        // as the rmsnorm sibling below — `a`/`b` arrive as strided
        // narrows of the fused in-proj output in the batched
        // concurrent decode path. `.contiguous()` is a no-op on
        // already-contig (rowwise) inputs.
        let a_c = a
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_qk_norm a contiguous failed")?;
        let b_c = b
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_qk_norm b contiguous failed")?;
        let alog_c = a_log
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_qk_norm a_log contiguous failed")?;
        let dtb_c = dt_bias
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_qk_norm dt_bias contiguous failed")?;
        let state_c = state
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_qk_norm state contiguous failed")?;
        if !kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent_supports_kt(
            q, k, v, &a_c, &b_c, &alog_c, &dtb_c, &state_c,
        ) {
            return Ok(None);
        }
        // Squeeze for the kernel call (3D-expecting).
        let q_3d = q
            .squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm q squeeze(1) failed")?;
        let k_3d = k
            .squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm k squeeze(1) failed")?;
        let v_3d = v
            .squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm v squeeze(1) failed")?;
        let out_kt = kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent_bf16_kt(
            &q_3d,
            &k_3d,
            &v_3d,
            &a_c,
            &b_c,
            &alog_c,
            &dtb_c,
            &state_c,
            q_scale as f32,
            qk_eps as f32,
        )
        .map_err(|e| anyhow::anyhow!("kt gdn_decode_qk_norm_gates_recurrent: {e}"))?;
        // Same 3D->4D unsqueeze fix as gdn_decode_gates_recurrent
        // above. The kt_api allocates 3D `[B, value_heads, dv]`
        // (see crates/kiln-gdn-kernel/src/kt_api.rs) but the trait
        // contract is 4D `[B, 1, value_heads, dv]`. (#1082)
        let out = out_kt
            .unsqueeze(1)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm out 3D->4D unsqueeze failed")?;
        Ok(Some(out))
    }

    #[allow(clippy::too_many_arguments)]
    fn runtime_gdn_decode_qk_norm_gates_recurrent_rmsnorm(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        q_scale: f64,
        qk_eps: f64,
        rms_eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !self.gdn_decode_qk_norm_recurrent_rmsnorm_enabled {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed bf16 surface is now the only path.
        // Args are already kt (#1082 DoD-101/102), so no candle↔kt
        // bridge. Non-bf16 envelopes return Ok(None) so the caller's
        // split gated_norm fallback engages. Production decode for
        // Qwen3.5-4B uses bf16 for all 10 input tensors (with F32
        // weight); the bf16_kt variant is the matching production hot
        // path.
        if !(q.dtype() == kiln_tensor::DType::BF16
            && k.dtype() == kiln_tensor::DType::BF16
            && v.dtype() == kiln_tensor::DType::BF16
            && a.dtype() == kiln_tensor::DType::BF16
            && b.dtype() == kiln_tensor::DType::BF16
            && a_log.dtype() == kiln_tensor::DType::BF16
            && dt_bias.dtype() == kiln_tensor::DType::BF16
            && state.dtype() == kiln_tensor::DType::BF16
            && z.dtype() == kiln_tensor::DType::BF16
            && weight.dtype() == kiln_tensor::DType::F32)
        {
            tracing::debug!(
                q_shape = ?q.shape(), q_dtype = ?q.dtype(),
                k_shape = ?k.shape(), k_dtype = ?k.dtype(),
                v_shape = ?v.shape(), v_dtype = ?v.dtype(),
                a_shape = ?a.shape(), a_dtype = ?a.dtype(),
                b_shape = ?b.shape(), b_dtype = ?b.dtype(),
                a_log_shape = ?a_log.shape(), a_log_dtype = ?a_log.dtype(),
                dt_bias_shape = ?dt_bias.shape(), dt_bias_dtype = ?dt_bias.dtype(),
                state_shape = ?state.shape(), state_dtype = ?state.dtype(), state_contiguous = state.is_contiguous(),
                z_shape = ?z.shape(), z_dtype = ?z.dtype(),
                weight_shape = ?weight.shape(), weight_dtype = ?weight.dtype(),
                "CUDA gdn_decode_qk_norm_gates_recurrent_rmsnorm declined (non-bf16 envelope); will retry with cast"
            );
            // Phase 5 fix (#1082): cast small 1-D weight tensors
            // (a_log, dt_bias) to BF16 + group-norm weight to F32
            // if needed. Same template as the sibling functions
            // above. Heavier tensors (q/k/v/a/b/state/z) shouldn't
            // be non-bf16 on the production hot path; if they are,
            // decline.
            let a_log_bf16 = if a_log.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    a_log
                        .to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_rmsnorm: cast a_log -> bf16")?,
                )
            };
            let dt_bias_bf16 = if dt_bias.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    dt_bias
                        .to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_rmsnorm: cast dt_bias -> bf16")?,
                )
            };
            let weight_f32 = if weight.dtype() == kiln_tensor::DType::F32 {
                None
            } else {
                Some(
                    weight
                        .to_dtype(kiln_tensor::DType::F32)
                        .with_context(|| "gdn_decode_rmsnorm: cast weight -> f32")?,
                )
            };
            // Same heavy-tensor cast as the sibling functions
            // above — required because conv1d emits F32 q/k/v.
            if state.dtype() != kiln_tensor::DType::BF16 {
                return Ok(None);
            }
            let q_bf16 = if q.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    q.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_rmsnorm: cast q -> bf16")?,
                )
            };
            let k_bf16 = if k.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    k.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_rmsnorm: cast k -> bf16")?,
                )
            };
            let v_bf16 = if v.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    v.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_rmsnorm: cast v -> bf16")?,
                )
            };
            let a_bf16 = if a.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    a.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_rmsnorm: cast a -> bf16")?,
                )
            };
            let b_bf16 = if b.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    b.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_rmsnorm: cast b -> bf16")?,
                )
            };
            let z_bf16 = if z.dtype() == kiln_tensor::DType::BF16 {
                None
            } else {
                Some(
                    z.to_dtype(kiln_tensor::DType::BF16)
                        .with_context(|| "gdn_decode_rmsnorm: cast z -> bf16")?,
                )
            };
            return self.runtime_gdn_decode_qk_norm_gates_recurrent_rmsnorm(
                q_bf16.as_ref().unwrap_or(q),
                k_bf16.as_ref().unwrap_or(k),
                v_bf16.as_ref().unwrap_or(v),
                a_bf16.as_ref().unwrap_or(a),
                b_bf16.as_ref().unwrap_or(b),
                a_log_bf16.as_ref().unwrap_or(a_log),
                dt_bias_bf16.as_ref().unwrap_or(dt_bias),
                state,
                z_bf16.as_ref().unwrap_or(z),
                weight_f32.as_ref().unwrap_or(weight),
                q_scale,
                qk_eps,
                rms_eps,
            );
        }
        kiln_nvtx::range!(c"kiln/gdn_decode_qk_norm_gates_recurrent_rmsnorm_bf16_kt");
        // supports_kt on 4D, kernel on 3D — see gdn_decode_gates_recurrent
        // above for the shape-contract split.
        //
        // #1082 bench regression (2026-05-26): in the *batched
        // concurrent decode* path, `a`/`b` arrive as
        // `ab.narrow(2, .., nv)` views on the fused A/B in-proj
        // output, producing non-contiguous last-dim views. Same
        // upstream shape as the gdn_gates path (line ~1574), which
        // already handles it via unconditional `.contiguous()`.
        // Without these calls every concurrent request ≥2 returned
        // HTTP 500 with "tensor must be contiguous" from
        // `kt_tensor_from_candle_cuda_borrow`. `.contiguous()` is a
        // no-op when the upstream tensor is already contiguous (the
        // rowwise bs=1 path).
        let a_c = a
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm a contiguous failed")?;
        let b_c = b
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm b contiguous failed")?;
        let alog_c = a_log
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm a_log contiguous failed")?;
        let dtb_c = dt_bias
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm dt_bias contiguous failed")?;
        let state_c = state
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm state contiguous failed")?;
        let z_c = z
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm z contiguous failed")?;
        let weight_c = weight
            .contiguous()
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm weight contiguous failed")?;
        if !kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent_rmsnorm_supports_kt(
            q, k, v, &a_c, &b_c, &alog_c, &dtb_c, &state_c, &z_c, &weight_c,
        ) {
            return Ok(None);
        }
        // Squeeze for the kernel call (3D-expecting).
        let q_3d = q
            .squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm q squeeze(1) failed")?;
        let k_3d = k
            .squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm k squeeze(1) failed")?;
        let v_3d = v
            .squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm v squeeze(1) failed")?;
        let out_kt = kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent_rmsnorm_bf16_kt(
            &q_3d,
            &k_3d,
            &v_3d,
            &a_c,
            &b_c,
            &alog_c,
            &dtb_c,
            &state_c,
            &z_c,
            &weight_c,
            q_scale as f32,
            qk_eps as f32,
            rms_eps as f32,
        )
        .map_err(|e| anyhow::anyhow!("kt gdn_decode_qk_norm_gates_recurrent_rmsnorm: {e}"))?;
        // Same 3D->4D unsqueeze fix as the gdn_decode_gates_recurrent
        // and gdn_decode_qk_norm_gates_recurrent wires above. The
        // kt_api allocates 3D `[B, value_heads, dv]` but the trait
        // contract is 4D `[B, 1, value_heads, dv]`. (#1082)
        let out = out_kt
            .unsqueeze(1)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm out 3D->4D unsqueeze failed")?;
        Ok(Some(out))
    }

    fn runtime_gdn_ab_in_proj_prefill(
        &self,
        x: &kiln_tensor::Tensor,
        in_proj_ab_t: &kiln_tensor::Tensor,
        nv: usize,
        seq_len: usize,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        let Ok((_, actual_seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        let Some(ab_dim) = nv.checked_mul(2) else {
            return Ok(None);
        };
        if !self.gdn_ab_in_proj_enabled
            || (seq_len != 1
                && (seq_len > CUDA_GDN_PREFILL_AB_IN_PROJ_MAX_TOKENS
                    || !self.gdn_prefill_ab_in_proj_enabled))
            || actual_seq_len != seq_len
            || seq_len == 0
            || x.dtype() != kiln_tensor::DType::BF16
            || in_proj_ab_t.dtype() != kiln_tensor::DType::BF16
            || in_proj_ab_t.track_op()
            || !in_proj_ab_t.is_contiguous()
            || in_proj_ab_t.dims() != [hidden, ab_dim]
            || !cuda_tensors_on_device(&[x, in_proj_ab_t])
        {
            return Ok(None);
        }

        let Some(ab) = LinearBackend::runtime_linear_prefill_apply(self, x, in_proj_ab_t)? else {
            return Ok(None);
        };
        let a = ab.narrow(2, 0, nv)?;
        let b = ab.narrow(2, nv, nv)?;
        Ok(Some((ab, a, b)))
    }

    fn runtime_supports_gdn_gates(&self) -> bool {
        self.support_predicates().supports_gdn_gates()
    }

    fn runtime_gdn_gates(
        &self,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        let dims = a.dims();
        let is_t1_decode = dims.len() >= 2 && dims[dims.len() - 2] == 1;
        if !is_t1_decode && !self.gdn_prefill_gates_enabled {
            tracing::debug!(
                a_shape = ?a.shape(),
                a_log_dtype = ?a_log.dtype(),
                dt_bias_dtype = ?dt_bias.dtype(),
                "CUDA prefill gdn_gates disabled; using Candle fallback"
            );
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed bf16 surface is now the only path.
        // Args are already kt (#1082 DoD-101/102), so no candle↔kt
        // bridge. Non-bf16 envelopes (f32/f32, f32/bf16) return
        // Ok(None) so the caller's candle fallback engages. bf16 was
        // the only envelope on the production decode/prefill path for
        // Qwen3.5-4B GDN.
        if !(a.dtype() == kiln_tensor::DType::BF16
            && b.dtype() == kiln_tensor::DType::BF16
            && a_log.dtype() == kiln_tensor::DType::BF16
            && dt_bias.dtype() == kiln_tensor::DType::BF16)
        {
            return Ok(None);
        }
        kiln_nvtx::range!(c"kiln/gdn_gates_bf16_kt");
        // At the bs>1 / prefill GDN call site, `a` and `b` arrive as
        // `ab.narrow(2, .., nv)` views on a fused A/B in-proj output,
        // which are non-contiguous on the last dim. The kt kernel
        // requires contiguous inputs, so we make each operand
        // contiguous here unconditionally. This is a no-op when the
        // upstream tensor is already contiguous (the seq_len==1 decode
        // case). a_log and dt_bias are weight tensors and are already
        // contiguous; the calls are kept for symmetry / future-proofing.
        let a_c = a
            .contiguous()
            .with_context(|| "kt-adapter: gdn_gates a contiguous failed")?;
        let b_c = b
            .contiguous()
            .with_context(|| "kt-adapter: gdn_gates b contiguous failed")?;
        let alog_c = a_log
            .contiguous()
            .with_context(|| "kt-adapter: gdn_gates a_log contiguous failed")?;
        let dtb_c = dt_bias
            .contiguous()
            .with_context(|| "kt-adapter: gdn_gates dt_bias contiguous failed")?;
        if !kiln_gdn_kernel::gdn_gates_supports_kt(&a_c, &b_c, &alog_c, &dtb_c) {
            return Ok(None);
        }
        let (beta_kt, g_kt) = kiln_gdn_kernel::gdn_gates_bf16_kt(&a_c, &b_c, &alog_c, &dtb_c)
            .map_err(|e| anyhow::anyhow!("kt gdn_gates_bf16: {e}"))?;
        Ok(Some((beta_kt, g_kt)))
    }

    fn runtime_supports_gdn_gated_rms_norm(&self) -> bool {
        self.support_predicates().supports_gdn_gated_rms_norm()
    }

    fn runtime_gdn_gated_rms_norm(
        &self,
        x: &kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !self.gdn_gated_rms_norm_enabled {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed bf16 activation surface. Qwen3.5 GDN
        // stores the learned RMSNorm scale as F32 in production, so support
        // both BF16 and F32 weight while keeping BF16 activations.
        if !(x.dtype() == kiln_tensor::DType::BF16
            && z.dtype() == kiln_tensor::DType::BF16
            && matches!(
                weight.dtype(),
                kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
            ))
        {
            return Ok(None);
        }
        kiln_nvtx::range!(c"kiln/gdn_gated_rms_norm_kt");
        // The kt variant expects rank-2 [rows, hidden]; flatten higher-rank
        // x/z by folding all leading dims into rows. weight stays [hidden].
        let x_dims = x.dims().to_vec();
        let hidden = *x_dims
            .last()
            .expect("x has at least one dim (checked by supports_kt)");
        let rows: usize = x_dims.iter().take(x_dims.len() - 1).product();
        // #1082: x/z arrive as transposed views (e.g. [1,7,2,16] strides
        // [224,16,112,1] after the head transpose). candle's reshape silently
        // copied; kt's reshape requires contiguous (and logs the copy), so
        // contiguify explicitly first — same materialization candle did.
        let x_flat = x
            .contiguous()
            .context("kt-adapter: gdn_gated_rms_norm contiguous x failed")?
            .reshape((rows, hidden))
            .context("kt-adapter: gdn_gated_rms_norm reshape x → [rows, hidden] failed")?;
        let z_flat = z
            .contiguous()
            .context("kt-adapter: gdn_gated_rms_norm contiguous z failed")?
            .reshape((rows, hidden))
            .context("kt-adapter: gdn_gated_rms_norm reshape z → [rows, hidden] failed")?;
        let out_kt = match weight.dtype() {
            kiln_tensor::DType::BF16 => {
                if !kiln_gdn_kernel::gdn_gated_rms_norm_supports_kt(&x_flat, &z_flat, weight) {
                    return Ok(None);
                }
                kiln_gdn_kernel::gdn_gated_rms_norm_bf16_kt(&x_flat, &z_flat, weight, eps as f32)
                    .map_err(|e| anyhow::anyhow!("kt gdn_gated_rms_norm_bf16: {e}"))?
            }
            kiln_tensor::DType::F32 => {
                if !kiln_gdn_kernel::gdn_gated_rms_norm_f32_weight_supports_kt(
                    &x_flat, &z_flat, weight,
                ) {
                    return Ok(None);
                }
                kiln_gdn_kernel::gdn_gated_rms_norm_bf16_f32_weight_kt(
                    &x_flat, &z_flat, weight, eps as f32,
                )
                .map_err(|e| anyhow::anyhow!("kt gdn_gated_rms_norm_bf16_f32_weight: {e}"))?
            }
            _ => return Ok(None),
        };
        let out = out_kt
            .reshape(x_dims)
            .context("kt-adapter: gdn_gated_rms_norm reshape out → original failed")?;
        Ok(Some(out))
    }
}

fn cuda_resident_activation_resource(
    tensor: &kiln_tensor::Tensor,
    state: super::residency::ResidentResourceState,
) -> super::residency::ResidentResource {
    super::residency::ResidentResource::from_tensor_for_backend(
        tensor,
        super::residency::resident_backend_for_runtime("cuda", tensor.device()),
        super::residency::ResidentResourceFamily::Activation,
        super::residency::ResidentOwnership::StorageOwned,
    )
    .with_state(state)
    .with_replay_stability(super::residency::ReplayStability::StableWithinStep)
}

impl super::residency::ResidentRegistry for CudaBackend {
    fn register_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
    ) -> Result<Option<super::residency::ResidentResource>> {
        if family != super::residency::ResidentResourceFamily::Activation {
            return Ok(None);
        }
        let resource = cuda_resident_activation_resource(
            tensor,
            super::residency::ResidentResourceState::RegisteredClean,
        );
        Ok(Some(super::cuda_rocm_common::mark_resident_activation(
            &self.resident_tensor_ids,
            tensor,
            resource,
            CUDA_RESIDENT_TENSOR_IDS_POISONED,
        )))
    }

    fn update_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
    ) -> Result<Option<super::residency::ResidentResource>> {
        if family != super::residency::ResidentResourceFamily::Activation {
            return Ok(None);
        }
        let resource = cuda_resident_activation_resource(
            tensor,
            super::residency::ResidentResourceState::DirtyDevice,
        );
        Ok(Some(super::cuda_rocm_common::mark_resident_activation(
            &self.resident_tensor_ids,
            tensor,
            resource,
            CUDA_RESIDENT_TENSOR_IDS_POISONED,
        )))
    }

    fn evict_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
    ) {
        if family != super::residency::ResidentResourceFamily::Activation {
            return;
        }
        super::cuda_rocm_common::evict_resident_activation(
            &self.resident_tensor_ids,
            tensor,
            CUDA_RESIDENT_TENSOR_IDS_POISONED,
        );
    }

    fn resident_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
    ) -> Option<super::residency::ResidentResource> {
        if family != super::residency::ResidentResourceFamily::Activation {
            return None;
        }
        super::cuda_rocm_common::resident_activation_resource(
            &self.resident_tensor_ids,
            tensor,
            CUDA_RESIDENT_TENSOR_IDS_POISONED,
        )
    }
}

impl ResidencyBackend for CudaBackend {
    fn runtime_supports_resident_activation(&self) -> bool {
        true
    }
}

impl TrainingLossBackend for CudaBackend {
    fn runtime_training_capabilities(&self) -> TrainingCapabilities {
        Self::training_capabilities_static()
    }

    fn runtime_training_precision_policy(&self) -> TrainingPrecisionPolicy {
        TrainingPrecisionPolicy::cuda()
    }
}

impl BackendRuntime for CudaBackend {}

#[allow(clippy::too_many_arguments)]
impl ReplayBackend for CudaBackend {
    fn runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        seqused_k: &kiln_tensor::Tensor,
        graph_outputs: Option<(&kiln_tensor::Tensor, &kiln_tensor::Tensor)>,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if q.track_op()
            || k_pool.track_op()
            || v_pool.track_op()
            || block_table.track_op()
            || seqused_k.track_op()
            || q.dtype() != kiln_tensor::DType::BF16
        {
            return Ok(None);
        }
        // Phase 7 opt-in (#1082): route through the kt-typed surface
        // only when there are no caller-owned graph outputs to write
        // through to. The kt-typed
        // `flash_attn_paged_decode_dyn_seqlen_kt` does not (yet)
        // accept a caller-owned (out, lse) pair — it allocates them
        // internally through the bridge — so the `graph_outputs ==
        // Some` case must stay on the candle path. That path
        // specifically exists to fix the dangling-pointer hazard
        // documented in
        // `bench-results/cuda-graph-bs2-secondary-audit.md` suspects
        // 3+4, where the CUDA graph runner re-uses caller-owned
        // tensors across replays. When `graph_outputs == None` (the
        // non-graph-capture path), the kt route is bit-exactly
        // equivalent because both paths bottom out in the same
        // `kiln_flash_attn_fwd_paged_decode_dyn_seqlen` FFI symbol.
        // Phase 7 (#1082): kt-only on both branches now that the
        // kt-typed `flash_attn_paged_decode_dyn_seqlen_kt_with_graph_
        // outputs` sibling exists. Bit-exact: both bottom out in the
        // same `kiln_flash_attn_fwd_paged_decode_dyn_seqlen` FFI
        // symbol. The caller-owned-output path is the
        // CUDA-graph-capture contract (the kernel writes through the
        // caller's pinned `(out, lse)` pair so graph replays don't
        // dangle on freshly-allocated scratch).
        // Phase 7 (#1082): args are already kt (#1082 DoD-101/102), so
        // no candle↔kt bridge — the kernel runs directly on the
        // caller's kt tensors.
        kiln_nvtx::range!(c"kiln/flash_attn_paged_decode_dyn_seqlen_kt");
        if let Some((out, lse)) = graph_outputs {
            // Caller owns `(out, lse)` (kt tensors). The kernel writes
            // in place via the with_graph_outputs kt entry; the
            // returned `out` is the caller's kt tensor whose CUDA
            // buffer the kernel mutated.
            kiln_flash_attn::flash_attn_paged_decode_dyn_seqlen_kt_with_graph_outputs(
                q,
                k_pool,
                v_pool,
                block_table,
                seqused_k,
                out,
                lse,
                max_seqlen_k,
                page_block_size,
                softmax_scale,
                causal,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "flash_attn_paged_decode_dyn_seqlen (graph variant) kt with_graph_outputs: {e}"
                )
            })?;
            return Ok(Some(out.clone()));
        }
        // No caller-owned outputs — allocate internally.
        let out_kt = kiln_flash_attn::flash_attn_paged_decode_dyn_seqlen_kt(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
        .map_err(|e| {
            anyhow::anyhow!("flash_attn_paged_decode_dyn_seqlen (graph variant) kt: {e}")
        })?;
        Ok(Some(out_kt))
    }
}

#[allow(clippy::too_many_arguments)]
impl LinearBackend for CudaBackend {
    fn runtime_supports_matmul_request(
        &self,
        req: &super::capability::MatmulRequest,
    ) -> super::capability::Support {
        let Some(rank) = matmul_request_support_rank(req) else {
            return super::capability::Support::Unsupported;
        };
        matmul_support_from_native(match req.epilogue {
            super::capability::MatmulEpilogue::Identity => true,
            super::capability::MatmulEpilogue::Bias => rank == 2,
            _ => false,
        })
    }

    fn runtime_matmul(
        &self,
        req: &super::capability::MatmulRequest,
        lhs: &kiln_tensor::Tensor,
        rhs: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !cuda_tensors_on_device(&[lhs, rhs])
            || !lhs.is_contiguous()
            || !rhs.is_contiguous()
            || req.out_dtype != lhs.dtype()
            || req.lhs_dtype != req.rhs_dtype
            || !matches!(
                lhs.dtype(),
                kiln_tensor::DType::BF16 | kiln_tensor::DType::F16 | kiln_tensor::DType::F32
            )
        {
            return Ok(None);
        }

        let Some(layout) = requested_matmul_layout(req, lhs, rhs) else {
            return Ok(None);
        };
        let out = match layout {
            BackendMatmulLayout::Plain => kiln_tensor::cuda_matmul(lhs, rhs)?,
            BackendMatmulLayout::LhsTransposed => {
                kiln_tensor::cuda_matmul_lhs_transposed(lhs, rhs)?
            }
            BackendMatmulLayout::RhsTransposed => {
                kiln_tensor::cuda_matmul_rhs_transposed(lhs, rhs)?
            }
            BackendMatmulLayout::BothTransposed => {
                let rank = lhs.rank();
                let lhs_t = lhs.transpose(rank - 2, rank - 1)?.contiguous()?;
                let rhs_t = rhs.transpose(rank - 2, rank - 1)?.contiguous()?;
                kiln_tensor::cuda_matmul(&lhs_t, &rhs_t)?
            }
        };
        Ok(Some(out))
    }

    fn runtime_lora_decode_add(
        &self,
        base: &kiln_tensor::Tensor,
        x: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !self.lora_decode_add_enabled
            || base.track_op()
            || x.track_op()
            || a.track_op()
            || b.track_op()
        {
            return Ok(None);
        }
        // #1082: args are already kt (BackendRuntime trait flipped to kt),
        // so there is no candle↔kt bridge — the kernel runs directly on the
        // caller's kt tensors and returns a kt result (zero candle roundtrip).
        if !kiln_rmsnorm_kernel::supports_lora_decode_add_kt(base, x, a, b) {
            return Ok(None);
        }
        let out_kt = kiln_rmsnorm_kernel::lora_decode_add_full_kt(base, x, a, b, scale)
            .map_err(|e| anyhow::anyhow!("kt lora_decode_add: {e}"))?;
        Ok(Some(out_kt))
    }

    fn runtime_linear_prefill_apply(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !cuda_tensors_on_device(&[x, weight_t])
            || x.dims().is_empty()
            || weight_t.dims().len() != 2
            || *x.dims().last().unwrap() != weight_t.dims()[0]
        {
            return Ok(None);
        }

        static FIRST_CUDA_LINEAR_PREFILL_LOGGED: OnceLock<()> = OnceLock::new();
        FIRST_CUDA_LINEAR_PREFILL_LOGGED.get_or_init(|| {
            tracing::info!(
                x_shape = ?x.dims(),
                weight_t_shape = ?weight_t.dims(),
                tracked = x.track_op() || weight_t.track_op(),
                "CudaBackend::linear_prefill_apply first call (kt cublasLt; tape records bwd)"
            );
        });

        // #1082: args are already kt. The kt matmul records onto the
        // autograd tape so `Tape::backward()` produces the projection
        // gradient — there is no candle autograd / CustomOp1 anymore.
        //
        // candle's `broadcast_matmul` for `[B, T, K] @ [K, N]` materializes
        // the broadcasted RHS via `.broadcast_as(...).contiguous()`, which on
        // CUDA copies the entire (B × K × N) weight tensor across the batch
        // dim before every matmul. nsys showed that copy at 78 % of total
        // GPU time on the bs > 1 GDN-decode path because GDN runs four
        // in-proj matmuls per layer × 24 layers per step and each pays a
        // ~168 MB BF16 copy at bs=4. Flatten leading dims, do a plain 2D
        // matmul, and reshape — same compute, no implicit contiguous copy.
        let l_dims = x.dims().to_vec();
        let k = l_dims[l_dims.len() - 1];
        let out_n = weight_t.dims()[1];
        let lead: usize = l_dims[..l_dims.len() - 1].iter().product();
        let out = if x.is_contiguous() {
            let x2d = x.reshape((lead, k))?;

            // Phase 7 opt-in: route through the kt cublasLt handle
            // when accelerator.kt_api_mode = "all" and the dtype is supported.
            // NVTX range from try_kt_matmul brackets the call as
            // kiln/matmul_kt in nsys.
            if crate::kt_api_policy::experimental_routes_enabled()
                && matches!(
                    x2d.dtype(),
                    kiln_tensor::DType::BF16 | kiln_tensor::DType::F16 | kiln_tensor::DType::F32
                )
                && x2d.dtype() == weight_t.dtype()
                && x2d.is_contiguous()
                && weight_t.is_contiguous()
            {
                if let Some(kt_out2d) = crate::forward::try_kt_matmul(&x2d, weight_t)? {
                    let mut out_shape = l_dims[..l_dims.len() - 1].to_vec();
                    out_shape.push(out_n);
                    CUDA_LINEAR_PREFILL_SUCCESSES.fetch_add(1, Ordering::Relaxed);
                    return Ok(Some(kt_out2d.reshape(out_shape)?));
                }
            }

            let out2d = x2d.matmul(weight_t)?;
            let mut out_shape = l_dims[..l_dims.len() - 1].to_vec();
            out_shape.push(out_n);
            out2d.reshape(out_shape)?
        } else {
            // Non-contiguous x fallback. broadcast_matmul materializes a
            // broadcasted-RHS copy internally (78 % GPU time at bs=4 due to
            // the per-step RHS copy). When the kt-API matmul gate is on and
            // dtypes line up, force x contiguous and route through cublasLt —
            // same 2D kt path as the is_contiguous() branch above, paying one
            // extra dtod for the x.contiguous() up front. That copy is bounded
            // by (B × T × K), whereas broadcast_matmul's implicit copy scales
            // as (B × K × N), typically larger by an order of magnitude on
            // Qwen3.5-4B GDN in-proj shapes. (#1082)
            if crate::kt_api_policy::experimental_routes_enabled()
                && matches!(
                    x.dtype(),
                    kiln_tensor::DType::BF16 | kiln_tensor::DType::F16 | kiln_tensor::DType::F32
                )
                && x.dtype() == weight_t.dtype()
                && weight_t.is_contiguous()
            {
                let x_c = x
                    .contiguous()
                    .context("linear_prefill_apply non-contig x: contiguous failed")?;
                let x2d = x_c.reshape((lead, k))?;
                if x2d.is_contiguous() {
                    if let Some(kt_out2d) = crate::forward::try_kt_matmul(&x2d, weight_t)? {
                        let mut out_shape = l_dims[..l_dims.len() - 1].to_vec();
                        out_shape.push(out_n);
                        CUDA_LINEAR_PREFILL_SUCCESSES.fetch_add(1, Ordering::Relaxed);
                        return Ok(Some(kt_out2d.reshape(out_shape)?));
                    }
                }
            }
            x.broadcast_matmul(weight_t)?
        };
        CUDA_LINEAR_PREFILL_SUCCESSES.fetch_add(1, Ordering::Relaxed);
        Ok(Some(out))
    }

    fn runtime_linear_prefill_apply_offset(
        &self,
        x: &kiln_tensor::Tensor,
        full_weight_t: &kiln_tensor::Tensor,
        chunk_start: usize,
        chunk_len: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !cuda_tensors_on_device(&[x, full_weight_t])
            || full_weight_t.dims().len() != 2
            || chunk_len == 0
            || chunk_start >= full_weight_t.dims()[1]
            || chunk_start + chunk_len > full_weight_t.dims()[1]
        {
            return Ok(None);
        }
        let chunk = full_weight_t
            .narrow(1, chunk_start, chunk_len)
            .context("cuda linear_prefill_apply_offset narrow weight chunk")?
            .contiguous()
            .context("cuda linear_prefill_apply_offset contiguous weight chunk")?;
        let chunk = if chunk.dtype() == x.dtype() {
            chunk
        } else {
            chunk
                .to_dtype(x.dtype())
                .context("cuda linear_prefill_apply_offset cast weight chunk")?
        };
        let out = self.runtime_linear_prefill_apply(x, &chunk)?;
        if out.is_some() {
            CUDA_LINEAR_PREFILL_OFFSET_SUCCESSES.fetch_add(1, Ordering::Relaxed);
        }
        Ok(out)
    }

    fn runtime_full_attn_qkv_combined_decode(
        &self,
        x: &kiln_tensor::Tensor,
        qkv_weight_t: Option<&kiln_tensor::Tensor>,
        _qkv_w8: Option<&crate::rocm_w8_proj::RocmW8Proj>,
        q_dim: usize,
        k_dim: usize,
        v_dim: usize,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        if !self.full_attn_qkv_in_proj_enabled
            || x.track_op()
            || x.dtype() != kiln_tensor::DType::BF16
        {
            return Ok(None);
        }
        let Some(qkv_weight_t) = qkv_weight_t else {
            return Ok(None);
        };
        let Ok((_, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        let qkv_dim = q_dim + k_dim + v_dim;
        if seq_len != 1
            || qkv_weight_t.dtype() != kiln_tensor::DType::BF16
            || qkv_weight_t.track_op()
            || !qkv_weight_t.is_contiguous()
            || qkv_weight_t.dims() != [hidden, qkv_dim]
            || !cuda_tensors_on_device(&[x, qkv_weight_t])
        {
            return Ok(None);
        }

        let Some(qkv) = self.runtime_linear_prefill_apply(x, qkv_weight_t)? else {
            return Ok(None);
        };
        let q_raw = qkv.narrow(2, 0, q_dim)?;
        let k_raw = qkv.narrow(2, q_dim, k_dim)?;
        let v = qkv.narrow(2, q_dim + k_dim, v_dim)?;
        Ok(Some((q_raw, k_raw, v)))
    }

    fn runtime_lora_delta_resident(
        &self,
        x: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !cuda_tensors_on_device(&[x, a, b])
            || !ResidencyBackend::runtime_has_resident_activation(self, a)
            || !ResidencyBackend::runtime_has_resident_activation(self, b)
        {
            return Ok(None);
        }

        // #1082: args are already kt. `compute_lora_delta` is kt-native and
        // records the delta matmul chain onto the autograd tape, so
        // `Tape::backward()` produces grad_A / grad_B — no candle autograd.
        let proj = LoraProjectionWeights {
            a: a.clone(),
            b: b.clone(),
        };
        let delta = compute_lora_delta(x, &proj, scale)
            .context("cuda registered LoRA delta (kt tape-recorded) failed")?;

        static FIRST_CUDA_LORA_DELTA_LOGGED: OnceLock<()> = OnceLock::new();
        FIRST_CUDA_LORA_DELTA_LOGGED.get_or_init(|| {
            tracing::info!(
                x_shape = ?x.dims(),
                a_shape = ?a.dims(),
                b_shape = ?b.dims(),
                scale,
                "CudaBackend::lora_delta_resident first call (kt tape-recorded)"
            );
        });

        Ok(Some(delta))
    }
}

#[allow(clippy::too_many_arguments)]
impl ConvBackend for CudaBackend {
    fn runtime_supports_causal_conv1d_update(&self) -> bool {
        self.support_predicates().supports_causal_conv1d_update()
    }

    fn runtime_supports_causal_conv1d_prefill(&self) -> bool {
        self.support_predicates().supports_causal_conv1d_prefill()
    }

    fn runtime_causal_conv1d_update(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !self.fused_conv1d_enabled {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed surface is now the only path; with
        // the BackendRuntime decode methods flipped to kt (#1082
        // DoD-101/102) the args are already kt, so the candle↔kt
        // bridges are gone — the kernel runs directly on the caller's
        // kt tensors. The kernel's in-place mutation of `conv_state`
        // surfaces through the caller's `&mut kt::Tensor` automatically
        // (anti-pattern 16 — owner-agnostic raw ptr).
        if !kiln_conv1d_kernel::supports_kt(x, weight, conv_state, kernel_size) {
            return Ok(None);
        }
        let out_kt =
            kiln_conv1d_kernel::causal_conv1d_update_kt(x, weight, conv_state, kernel_size)
                .map_err(|e| anyhow::anyhow!("kt causal_conv1d_update: {e}"))?;
        Ok(Some(out_kt))
    }

    fn runtime_causal_conv1d_prefill(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !self.fused_conv1d_enabled {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed surface is now the only path.
        // See update path above for the in-place semantics.
        if !kiln_conv1d_kernel::supports_prefill_kt(x, weight, conv_state, kernel_size) {
            return Ok(None);
        }
        let out_kt =
            kiln_conv1d_kernel::causal_conv1d_prefill_kt(x, weight, conv_state, kernel_size)
                .map_err(|e| anyhow::anyhow!("kt causal_conv1d_prefill: {e}"))?;
        Ok(Some(out_kt))
    }
}

impl SamplingBackend for CudaBackend {}

impl PagedKvBackend for CudaBackend {}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_backend() -> CudaBackend {
        // Test mock on a CPU kt device — the unit tests in this module only
        // exercise the kt-typed surface. (#1082 DoD-100 step 4: CudaBackend's
        // candle `device` field was dropped; `device_kt` is the sole device.)
        CudaBackend {
            device_kt: kiln_tensor::Device::Cpu,
            resident_tensor_ids: crate::backend::cuda_rocm_common::new_resident_tensor_id_registry(
            ),
            full_attn_qkv_in_proj_enabled: false,
            gdn_ab_in_proj_enabled: false,
            gdn_prefill_ab_in_proj_enabled: false,
            gdn_prefill_gates_enabled: false,
            gdn_enabled: false,
            gdn_gates_enabled: false,
            gdn_gated_rms_norm_enabled: false,
            gdn_decode_fused_enabled: false,
            gdn_decode_unexpanded_qk_enabled: false,
            gdn_decode_qk_norm_recurrent_enabled: false,
            gdn_decode_qk_norm_recurrent_rmsnorm_enabled: false,
            fused_conv1d_enabled: false,
            lora_decode_add_enabled: false,
            gdn_full_chunk_forward_multiblock_enabled: false,
        }
    }

    #[test]
    fn kernel_profiles_are_complete_backend_local_policies() {
        let routes = |backend: &CudaBackend| {
            [
                backend.full_attn_qkv_in_proj_enabled,
                backend.gdn_ab_in_proj_enabled,
                backend.gdn_prefill_ab_in_proj_enabled,
                backend.gdn_prefill_gates_enabled,
                backend.gdn_enabled,
                backend.gdn_gates_enabled,
                backend.gdn_gated_rms_norm_enabled,
                backend.gdn_decode_fused_enabled,
                backend.gdn_decode_unexpanded_qk_enabled,
                backend.gdn_decode_qk_norm_recurrent_enabled,
                backend.gdn_decode_qk_norm_recurrent_rmsnorm_enabled,
                backend.fused_conv1d_enabled,
                backend.lora_decode_add_enabled,
                backend.gdn_full_chunk_forward_multiblock_enabled,
            ]
        };
        let native_default = CudaBackend::new_with_kernel_policy(
            kiln_tensor::Device::Cuda(0),
            CudaKernelPolicy::native_default(),
        );
        assert_eq!(routes(&native_default), [true; 14]);

        let fallback = CudaBackend::new_with_kernel_policy(
            kiln_tensor::Device::Cuda(0),
            CudaKernelPolicy::portable_fallback(),
        );
        assert_eq!(routes(&fallback), [false; 14]);
    }

    #[test]
    fn cuda_resident_activation_registry_lifecycle() -> Result<()> {
        // #1082: BackendRuntime residency hooks are kt-typed; the registry
        // keys on the kt TensorId directly.
        use kiln_tensor::{DType as KtDType, Tensor as KtTensor};
        let backend = test_backend();
        let tensor = KtTensor::zeros_cpu(vec![2, 3], KtDType::F32);

        assert!(ResidencyBackend::runtime_supports_resident_activation(
            &backend
        ));
        assert!(!ResidencyBackend::runtime_has_resident_activation(
            &backend, &tensor
        ));

        ResidencyBackend::runtime_register_resident_activation(&backend, &tensor)?;
        assert!(ResidencyBackend::runtime_has_resident_activation(
            &backend, &tensor
        ));

        ResidencyBackend::runtime_evict_resident_activation(&backend, &tensor);
        assert!(!ResidencyBackend::runtime_has_resident_activation(
            &backend, &tensor
        ));

        ResidencyBackend::runtime_update_resident_activation(&backend, &tensor)?;
        assert!(ResidencyBackend::runtime_has_resident_activation(
            &backend, &tensor
        ));

        Ok(())
    }

    #[test]
    fn cuda_optimizer_dispatch_declines_without_cuda_tensors() -> Result<()> {
        // #1082: dispatch_{sgd,adamw}_step are kt-typed. CPU kt tensors must
        // be declined (the kernels only service CUDA-resident tensors).
        use kiln_tensor::{DType as KtDType, Tensor as KtTensor};
        let backend = test_backend();
        let param = KtTensor::zeros_cpu(vec![2, 3], KtDType::F32);
        let grad = KtTensor::ones(vec![2, 3], KtDType::F32, kiln_tensor::Device::Cpu)?;
        let m = KtTensor::zeros_cpu(vec![2, 3], KtDType::F32);
        let v = KtTensor::zeros_cpu(vec![2, 3], KtDType::F32);

        assert!(
            !OptimizerBackend::runtime_dispatch_sgd_step(&backend, &param, &grad, 0.01)?,
            "CUDA must not claim SGD dispatch for non-CUDA tensors"
        );
        assert!(
            !OptimizerBackend::runtime_dispatch_adamw_step(
                &backend, &param, &grad, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, 1
            )?,
            "CUDA must not claim AdamW dispatch for non-CUDA tensors"
        );

        ResidencyBackend::runtime_register_resident_activation(&backend, &param)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &grad)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &m)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &v)?;

        assert!(
            !OptimizerBackend::runtime_dispatch_sgd_step(&backend, &param, &grad, 0.01)?,
            "TensorId residency alone is not enough for CUDA to claim SGD ownership"
        );
        assert!(
            !OptimizerBackend::runtime_dispatch_adamw_step(
                &backend, &param, &grad, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, 1
            )?,
            "TensorId residency alone is not enough for CUDA to claim AdamW ownership"
        );

        Ok(())
    }

    /// #1082: build a CUDA-resident kt tensor from host data, or skip the
    /// test when no CUDA device is available. Returns `Ok(None)` to signal
    /// skip (mirrors the historical `candle_core::Device::new_cuda` match).
    fn kt_cuda_f32(values: &[f32]) -> Result<Option<kiln_tensor::Tensor>> {
        let host = kiln_tensor::Tensor::from_slice(values, vec![values.len()])?;
        match host.to_device(kiln_tensor::Device::Cuda(0)) {
            Ok(t) => Ok(Some(t)),
            Err(_) => Ok(None),
        }
    }

    #[test]
    fn cuda_sgd_step_resident_round_trip_f32() -> Result<()> {
        use kiln_tensor::Device as KtDevice;
        let backend = CudaBackend::new(kiln_tensor::Device::Cuda(0));
        let Some(param) = kt_cuda_f32(&[1.0f32, -2.0, 0.5, 3.0])? else {
            eprintln!("CUDA unavailable, skipping cuda_sgd_step_resident_round_trip_f32");
            return Ok(());
        };
        let grad = kt_cuda_f32(&[0.1f32, -0.2, 0.5, 1.0])?.expect("cuda grad");
        ResidencyBackend::runtime_register_resident_activation(&backend, &param)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &grad)?;

        assert!(OptimizerBackend::runtime_dispatch_sgd_step(
            &backend, &param, &grad, 0.25
        )?);
        let actual = param.to_device(KtDevice::Cpu)?.to_vec1::<f32>()?;
        let expected = [0.975f32, -1.95, 0.375, 2.75];
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < 1e-6,
                "actual={actual:?} expected={expected:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn cuda_adamw_step_resident_round_trip_f32() -> Result<()> {
        use kiln_tensor::Device as KtDevice;
        let backend = CudaBackend::new(kiln_tensor::Device::Cuda(0));
        let Some(param) = kt_cuda_f32(&[1.0f32, -2.0, 0.5, 3.0])? else {
            eprintln!("CUDA unavailable, skipping cuda_adamw_step_resident_round_trip_f32");
            return Ok(());
        };
        let grad = kt_cuda_f32(&[0.5f32, -0.5, 0.25, -0.25])?.expect("cuda grad");
        let m = kt_cuda_f32(&[0.0f32, 0.0, 0.0, 0.0])?.expect("cuda m");
        let v = kt_cuda_f32(&[0.0f32, 0.0, 0.0, 0.0])?.expect("cuda v");
        ResidencyBackend::runtime_register_resident_activation(&backend, &param)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &grad)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &m)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &v)?;

        let lr = 0.01;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let weight_decay = 0.1;
        assert!(OptimizerBackend::runtime_dispatch_adamw_step(
            &backend,
            &param,
            &grad,
            &m,
            &v,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            1,
        )?);

        let actual = param.to_device(KtDevice::Cpu)?.to_vec1::<f32>()?;
        let before = [1.0f32, -2.0, 0.5, 3.0];
        let grad_vals = [0.5f32, -0.5, 0.25, -0.25];
        for ((a, p0), g) in actual.iter().zip(before.iter()).zip(grad_vals.iter()) {
            let p_after_wd = *p0 * (1.0 - lr * weight_decay);
            let expected = p_after_wd - lr * (*g / (g.abs() + eps));
            assert!((a - expected).abs() < 1e-5, "actual={actual:?}");
        }
        let m_actual = m.to_device(KtDevice::Cpu)?.to_vec1::<f32>()?;
        let v_actual = v.to_device(KtDevice::Cpu)?.to_vec1::<f32>()?;
        for ((m_i, v_i), g) in m_actual.iter().zip(v_actual.iter()).zip(grad_vals.iter()) {
            assert!((m_i - (1.0 - beta1) * g).abs() < 1e-6);
            assert!((v_i - (1.0 - beta2) * g * g).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn cuda_sgd_and_adamw_resident_round_trip_bf16() -> Result<()> {
        use kiln_tensor::{DType as KtDType, Device as KtDevice};
        let backend = CudaBackend::new(kiln_tensor::Device::Cuda(0));

        // Build host BF16 then move to CUDA; skip when no device.
        let mk_bf16 = |vals: &[f32]| -> Result<Option<kiln_tensor::Tensor>> {
            let host =
                kiln_tensor::Tensor::from_slice(vals, vec![vals.len()])?.to_dtype(KtDType::BF16)?;
            match host.to_device(KtDevice::Cuda(0)) {
                Ok(t) => Ok(Some(t)),
                Err(_) => Ok(None),
            }
        };

        let Some(param) = mk_bf16(&[1.0f32, -2.0, 0.5, 3.0])? else {
            eprintln!("CUDA unavailable, skipping cuda_sgd_and_adamw_resident_round_trip_bf16");
            return Ok(());
        };
        let grad = mk_bf16(&[0.25f32, -0.5, 0.5, -0.25])?.expect("cuda grad");
        ResidencyBackend::runtime_register_resident_activation(&backend, &param)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &grad)?;
        assert!(OptimizerBackend::runtime_dispatch_sgd_step(
            &backend, &param, &grad, 0.5
        )?);
        let sgd_actual = param
            .to_device(KtDevice::Cpu)?
            .to_dtype(KtDType::F32)?
            .to_vec1::<f32>()?;
        let sgd_expected = [0.875f32, -1.75, 0.25, 3.125];
        for (a, e) in sgd_actual.iter().zip(sgd_expected.iter()) {
            assert!(
                (a - e).abs() < 0.02,
                "actual={sgd_actual:?} expected={sgd_expected:?}"
            );
        }

        let adam_param = mk_bf16(&[1.0f32, -2.0, 0.5, 3.0])?.expect("cuda adam_param");
        let adam_grad = mk_bf16(&[0.5f32, -0.5, 0.25, -0.25])?.expect("cuda adam_grad");
        let m = mk_bf16(&[0.0f32, 0.0, 0.0, 0.0])?.expect("cuda m");
        let v = mk_bf16(&[0.0f32, 0.0, 0.0, 0.0])?.expect("cuda v");
        ResidencyBackend::runtime_register_resident_activation(&backend, &adam_param)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &adam_grad)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &m)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &v)?;
        assert!(OptimizerBackend::runtime_dispatch_adamw_step(
            &backend,
            &adam_param,
            &adam_grad,
            &m,
            &v,
            0.01,
            0.9,
            0.999,
            1e-8,
            0.1,
            1,
        )?);
        let adam_actual = adam_param
            .to_device(KtDevice::Cpu)?
            .to_dtype(KtDType::F32)?
            .to_vec1::<f32>()?;
        let before = [1.0f32, -2.0, 0.5, 3.0];
        let grad_vals = [0.5f32, -0.5, 0.25, -0.25];
        for ((a, p0), g) in adam_actual.iter().zip(before.iter()).zip(grad_vals.iter()) {
            let expected = *p0 * (1.0 - 0.01 * 0.1) - 0.01 * (*g / (g.abs() + 1e-8));
            assert!((a - expected).abs() < 0.03, "actual={adam_actual:?}");
        }
        Ok(())
    }

    #[test]
    fn cuda_flash_attention_declines_unsupported_dtype() -> Result<()> {
        // #1082 DoD-101/102: the FlashAttention BackendRuntime methods
        // are now kt-typed. The kt forward path is detached
        // (track_op() is always false), so the historical
        // candle-`Var`-tracked decline gate no longer applies. The
        // production decline contract these methods still enforce is
        // the dtype gate: non-BF16 inputs return Ok(None) so the
        // caller falls back to the portable SDPA path. We exercise
        // that with F32 kt tensors here (the CPU test backend can't
        // dispatch the real CUDA kernel anyway).
        use kiln_tensor::{DType as KtDType, Tensor as KtTensor};
        let backend = test_backend();

        let q = KtTensor::zeros_cpu(vec![1, 2, 1, 128], KtDType::F32);
        let k = KtTensor::zeros_cpu(vec![1, 2, 1, 128], KtDType::F32);
        let v = KtTensor::zeros_cpu(vec![1, 2, 1, 128], KtDType::F32);
        assert!(
            backend
                .runtime_flash_attn_prefill(&q, &k, &v, 1.0, true)?
                .is_none(),
            "CUDA FlashAttention prefill must decline non-BF16 inputs"
        );

        let q_decode = KtTensor::zeros_cpu(vec![1, 1, 1, 128], KtDType::F32);
        let k_pool = KtTensor::zeros_cpu(vec![128, 1, 128], KtDType::F32);
        let v_pool = KtTensor::zeros_cpu(vec![128, 1, 128], KtDType::F32);
        let block_table = KtTensor::zeros_cpu(vec![1, 1], KtDType::U32);
        let seqused_k = KtTensor::zeros_cpu(vec![1], KtDType::I64);

        assert!(
            backend
                .runtime_flash_attn_paged_decode(
                    &q_decode,
                    &k_pool,
                    &v_pool,
                    &block_table,
                    128,
                    128,
                    1.0,
                    true
                )?
                .is_none(),
            "CUDA paged decode attention must decline non-BF16 inputs"
        );
        assert!(
            backend
                .runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
                    &q_decode,
                    &k_pool,
                    &v_pool,
                    &block_table,
                    &seqused_k,
                    128,
                    128,
                    1.0,
                    true,
                )?
                .is_none(),
            "CUDA dynamic paged decode attention must decline non-BF16 inputs"
        );

        Ok(())
    }

    /// #1082: build a CUDA-resident 2-D kt tensor from host data, or
    /// `Ok(None)` to skip when no CUDA device is present.
    fn kt_cuda_2d(values: &[f32], shape: [usize; 2]) -> Result<Option<kiln_tensor::Tensor>> {
        let host = kiln_tensor::Tensor::from_slice(values, vec![shape[0], shape[1]])?;
        match host.to_device(kiln_tensor::Device::Cuda(0)) {
            Ok(t) => Ok(Some(t)),
            Err(_) => Ok(None),
        }
    }

    #[test]
    fn cuda_linear_prefill_apply_matches_reference_matmul() -> Result<()> {
        let backend = CudaBackend::new(kiln_tensor::Device::Cuda(0));

        let Some(x) = kt_cuda_2d(&[1.0f32, -2.0, 0.5, 3.0, 4.0, -1.0], [2, 3])? else {
            eprintln!(
                "CUDA unavailable, skipping cuda_linear_prefill_apply_matches_reference_matmul"
            );
            return Ok(());
        };
        let w = kt_cuda_2d(
            &[
                0.5f32, 1.0, -1.5, 2.0, -0.25, 0.75, 1.25, -0.5, 2.0, -1.0, 0.0, 0.5,
            ],
            [3, 4],
        )?
        .expect("cuda w");

        let routed = LinearBackend::runtime_linear_prefill_apply(&backend, &x, &w)?
            .expect("CUDA linear_prefill_apply should accept CUDA tensors");
        let expected = x.broadcast_matmul(&w)?;
        assert_eq!(
            routed
                .to_device(kiln_tensor::Device::Cpu)?
                .to_vec2::<f32>()?,
            expected
                .to_device(kiln_tensor::Device::Cpu)?
                .to_vec2::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn cuda_linear_prefill_apply_offset_matches_reference_chunk() -> Result<()> {
        let backend = CudaBackend::new(kiln_tensor::Device::Cuda(0));

        let Some(x) = kt_cuda_2d(&[1.0f32, -2.0, 0.5, 3.0, 4.0, -1.0], [2, 3])? else {
            eprintln!(
                "CUDA unavailable, skipping cuda_linear_prefill_apply_offset_matches_reference_chunk"
            );
            return Ok(());
        };
        let w = kt_cuda_2d(
            &[
                0.5f32, 1.0, -1.5, 2.0, 3.0, -0.25, 0.75, 1.25, -0.5, 0.25, 2.0, -1.0, 0.0, 0.5,
                -2.0,
            ],
            [3, 5],
        )?
        .expect("cuda w");

        let routed = LinearBackend::runtime_linear_prefill_apply_offset(&backend, &x, &w, 1, 3)?
            .expect("CUDA linear_prefill_apply_offset should accept CUDA tensors");
        let expected_chunk = w.narrow(1, 1, 3)?.contiguous()?;
        let expected = x.broadcast_matmul(&expected_chunk)?;
        assert_eq!(
            routed
                .to_device(kiln_tensor::Device::Cpu)?
                .to_vec2::<f32>()?,
            expected
                .to_device(kiln_tensor::Device::Cpu)?
                .to_vec2::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn cuda_registered_lora_delta_matches_reference() -> Result<()> {
        let backend = CudaBackend::new(kiln_tensor::Device::Cuda(0));

        let Some(x) = kt_cuda_2d(&[0.5f32, -1.0, 2.0, 1.5, 0.25, -0.75], [2, 3])? else {
            eprintln!("CUDA unavailable, skipping cuda_registered_lora_delta_matches_reference");
            return Ok(());
        };
        let a = kt_cuda_2d(&[0.25f32, -0.5, 1.0, 1.5, 0.0, -1.0], [2, 3])?.expect("cuda a");
        let b =
            kt_cuda_2d(&[1.0f32, -0.25, 0.5, 0.75, -1.0, 0.25, 0.0, 1.5], [4, 2])?.expect("cuda b");
        let scale = 0.5;

        assert!(LinearBackend::runtime_lora_delta_resident(&backend, &x, &a, &b, scale)?.is_none());
        ResidencyBackend::runtime_register_resident_activation(&backend, &a)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &b)?;

        let routed = LinearBackend::runtime_lora_delta_resident(&backend, &x, &a, &b, scale)?
            .expect("registered CUDA LoRA delta should engage");
        let expected = compute_lora_delta(
            &x,
            &LoraProjectionWeights {
                a: a.clone(),
                b: b.clone(),
            },
            scale,
        )?;

        assert_eq!(
            routed
                .to_device(kiln_tensor::Device::Cpu)?
                .to_vec2::<f32>()?,
            expected
                .to_device(kiln_tensor::Device::Cpu)?
                .to_vec2::<f32>()?
        );
        Ok(())
    }
}
