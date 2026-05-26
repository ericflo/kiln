//! CUDA backend: FlashAttention-2 and Gated DeltaNet fused kernels.
//!
//! Wraps the vendored `kiln-flash-attn` and `kiln-gdn-kernel` crates.
//! `Ok(None)` responses route the caller to the portable candle path.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use super::{BackendRuntime, TrainingCapabilities};
use crate::lora_loader::{LoraProjectionWeights, compute_lora_delta};

static CUDA_RESIDENT_TENSOR_IDS: OnceLock<Mutex<HashSet<candle_core::TensorId>>> = OnceLock::new();
static CUDA_SGD_DISPATCH_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static CUDA_ADAMW_DISPATCH_SUCCESSES: AtomicU64 = AtomicU64::new(0);
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

pub fn flash_attn_tracked_decline_count() -> u64 {
    CUDA_FLASH_ATTN_TRACKED_DECLINES.load(Ordering::Relaxed)
}

pub fn reset_flash_attn_tracked_decline_count() {
    CUDA_FLASH_ATTN_TRACKED_DECLINES.store(0, Ordering::Relaxed);
}

/// `(multiblock_path_successes, single_block_path_successes)` for
/// `gdn_full_chunk_forward`. Used by tests and bench tooling to confirm which
/// kernel actually ran under a given env-var configuration.
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

fn any_tracks_op(tensors: &[&Tensor]) -> bool {
    tensors.iter().any(|tensor| tensor.track_op())
}

fn with_cuda_resident_ids<R>(f: impl FnOnce(&mut HashSet<candle_core::TensorId>) -> R) -> R {
    let registry = CUDA_RESIDENT_TENSOR_IDS.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = registry
        .lock()
        .expect("CUDA resident TensorId registry mutex poisoned");
    f(&mut guard)
}

#[derive(Debug)]
pub struct CudaBackend {
    device: Device,
    /// Cached at construction: reading env vars per decode step × 24 GDN layers
    /// shows up in decode NVTX captures. Env vars don't change at runtime.
    gdn_enabled: bool,
    /// Same pattern: cache the env-var read. The fused gates kernel is
    /// gated behind its own kill switch so it can be disabled independently.
    gdn_gates_enabled: bool,
    /// Kill switch for the fused GDN gated RMSNorm kernel (decode/prefill
    /// kiln/gdn/gated_norm region).
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
    /// Kill switch for the fused causal_conv1d_update kernel (decode
    /// kiln/gdn/conv region). When off, forward.rs falls back to the
    /// candle to_f32/cat/sum/narrow chain.
    fused_conv1d_enabled: bool,
    // Phase 7 (#1082): the cuda_use_kt_api_conv1d gate was removed once
    // the kt-typed surface (causal_conv1d_{update,prefill}_kt +
    // supports{,_prefill}_kt) became the only path. The escape hatch
    // for the conv kernel as a whole is still `fused_conv1d_enabled`
    // (KILN_DISABLE_FUSED_CONV1D), which falls back to forward.rs's
    // candle to_f32/cat/sum/narrow chain — the kt-typed path is bit-
    // exact with the previous kt-API code (same FFI symbol).
    // Phase 7 (#1082): the cuda_use_kt_api_gdn gate was removed once
    // all 10 GDN dispatch wires (forward_substitution, recurrent_step,
    // chunk_prep, chunk_scan, full_chunk_forward[_multiblock],
    // gates, gated_rms_norm, plus the 4 decode_* wires:
    // gates_recurrent, qk_norm_gates_recurrent,
    // qk_norm_gates_recurrent_rmsnorm) became kt-only. The whole-
    // kernel kill switch `KILN_DISABLE_GDN_KERNEL=1` plus the per-
    // wire decode-fused kill switches (KILN_DISABLE_FUSED_GDN_DECODE,
    // KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT[_RMSNORM]) still
    // fall back to forward.rs's candle reference paths. The kt-typed
    // path is bit-exact with the previous kt-API code (same FFI
    // symbol). One candle-typed caller survives: the single-block
    // `kiln_gdn_kernel::gdn_full_chunk_forward` fall-through inside
    // the multiblock dispatcher, because no kt single-block kernel
    // wire exists yet.
    // Phase 7 (#1082): the cuda_use_kt_api_flash_attn gate was
    // removed once the kt-typed surface became the only path on
    // the 3 sites that don't take caller-owned graph outputs
    // (`flash_attn`, `flash_attn_paged_decode`,
    // `flash_attn_paged_decode_contiguous_batch_dyn_seqlen`). The
    // 4th site (`_with_graph_outputs`) still routes the
    // `graph_outputs == Some(...)` case through the candle wrapper
    // because the kt-typed entry doesn't accept caller-owned (out,
    // lse) pairs yet; the `graph_outputs == None` case is kt.
    /// Phase 7 opt-in (#1082): route the SGD optimizer step through
    /// the kt-typed surface (`sgd_step_{f32,bf16}_kt`) instead of the
    /// candle-typed `kiln_rmsnorm_kernel::sgd_step_inplace` shim.
    /// Default OFF — flips on once per-call parity coverage lands.
    /// The kt path is bit-exact by construction: both paths bottom
    /// out in the same FFI symbols (`kiln_sgd_step_f32`,
    /// `kiln_sgd_step_bf16`); only the Rust shell types change. The
    /// in-place mutation surfaces in the caller's candle tensor via
    /// the zero-copy `kt_tensor_from_candle_cuda_borrow` adapter.
    /// Set `KILN_USE_KT_API_SGD_STEP=1` (or `KILN_USE_KT_API_ALL=1`)
    /// to enable.
    cuda_use_kt_api_sgd_step: bool,
    /// Phase 7 opt-in (#1082): route the AdamW optimizer step through
    /// the kt-typed surface (`adamw_step_{f32,bf16}_kt`) instead of
    /// the candle-typed `kiln_rmsnorm_kernel::adamw_step_inplace`
    /// shim. Default OFF — flips on once per-call parity coverage
    /// lands. Same bit-exact-by-construction rationale as
    /// `cuda_use_kt_api_sgd_step` (both bottom out in the same
    /// `kiln_adamw_step_{f32,bf16}` FFI symbols). The candle shim
    /// computes bias-correction terms internally from `step: u32`;
    /// the kt path takes pre-computed bias_correction1 /
    /// bias_correction2 instead, so we replicate the same
    /// `(1 - beta^step).max(1e-20)` formula at the caller. Set
    /// `KILN_USE_KT_API_ADAMW_STEP=1` (or `KILN_USE_KT_API_ALL=1`)
    /// to enable.
    cuda_use_kt_api_adamw_step: bool,
    /// Forward-only CUDA LoRA delta/add for decode. Training declines because
    /// tracked LoRA tensors need autograd.
    lora_decode_add_enabled: bool,
    /// Multi-block dv-tiled `gdn_full_chunk_forward`. Default ON because the
    /// single-block kernel only launches `B*H = 32` blocks for Qwen3.5-4B at
    /// batch=1, leaving ~58% of a 76-SM RTX 4090 Laptop idle. The multi-block
    /// path is bit-exact with the legacy kernel (same per-output-cell FMA
    /// chain, same bf16 rounding). Set
    /// `KILN_DISABLE_GDN_FULL_CHUNK_FORWARD_MULTIBLOCK=1` to fall back.
    gdn_full_chunk_forward_multiblock_enabled: bool,
}

impl CudaBackend {
    pub fn new(device: Device) -> Self {
        debug_assert!(device.is_cuda(), "CudaBackend created on non-CUDA device");
        let gdn_enabled = std::env::var("KILN_DISABLE_GDN_KERNEL").is_err();
        let gdn_gates_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_FUSED_GDN_GATES").is_err();
        let gdn_gated_rms_norm_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_FUSED_GDN_GATED_RMS_NORM").is_err();
        let fused_conv1d_enabled = std::env::var("KILN_DISABLE_FUSED_CONV1D").is_err();
        // #1082: the dedicated KILN_DISABLE_KT_API_CONV1D gate was
        // removed once the kt-typed conv1d surface became the only
        // path in `causal_conv1d_{update,prefill}`. The whole-kernel
        // kill switch `KILN_DISABLE_FUSED_CONV1D=1` still falls back
        // to forward.rs's candle to_f32/cat/sum/narrow chain.
        // #1082: the dedicated KILN_DISABLE_KT_API_GDN gate was
        // removed once the kt-typed GDN surface became the only path
        // across all 10 dispatch wires. The whole-kernel kill switch
        // `KILN_DISABLE_GDN_KERNEL=1` (plus the per-wire decode-fused
        // kill switches) still falls back to forward.rs's candle
        // reference paths.
        // #1082: flipped default ON. The kt path is bit-exact by
        // construction — all 4 wired flash_attn dispatch sites bottom
        // out in the same FFI symbols as the candle shim, with only
        // the Rust shell types changing. The `with_graph_outputs`
        // site retains its `graph_outputs.is_none()` guard so the
        // caller-owned-output path keeps using the candle wrapper.
        // KILN_DISABLE_KT_API_FLASH_ATTN gate removed alongside the
        // 3 sites where the kt-typed path is the only path. The
        // 4th site checks `graph_outputs.is_none()` directly.
        // #1082: opt-in (default off). The kt path is bit-exact by
        // construction — both candle and kt paths bottom out in
        // the same `kiln_sgd_step_{f32,bf16}` FFI symbols; only the
        // Rust shell types change. Flips on once per-call parity
        // coverage lands. Opt-in: `KILN_USE_KT_API_SGD_STEP=1` (or
        // `KILN_USE_KT_API_ALL=1`).
        let cuda_use_kt_api_sgd_step = std::env::var("KILN_USE_KT_API_SGD_STEP").is_ok()
            || std::env::var("KILN_USE_KT_API_ALL").is_ok();
        // #1082: opt-in (default off). Same bit-exact-by-construction
        // rationale as `cuda_use_kt_api_sgd_step` (both candle and
        // kt paths bottom out in the same `kiln_adamw_step_{f32,bf16}`
        // FFI symbols). Opt-in: `KILN_USE_KT_API_ADAMW_STEP=1` (or
        // `KILN_USE_KT_API_ALL=1`).
        let cuda_use_kt_api_adamw_step = std::env::var("KILN_USE_KT_API_ADAMW_STEP").is_ok()
            || std::env::var("KILN_USE_KT_API_ALL").is_ok();
        let gdn_decode_fused_enabled = gdn_gates_enabled
            && gdn_gated_rms_norm_enabled
            && std::env::var("KILN_DISABLE_FUSED_GDN_DECODE").is_err();
        let gdn_decode_unexpanded_qk_enabled = gdn_decode_fused_enabled
            && std::env::var("KILN_DISABLE_GDN_DECODE_UNEXPANDED_QK").is_err();
        let gdn_decode_qk_norm_recurrent_enabled = gdn_decode_unexpanded_qk_enabled
            && std::env::var("KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT").is_err();
        let gdn_decode_qk_norm_recurrent_rmsnorm_enabled = gdn_decode_qk_norm_recurrent_enabled
            && std::env::var("KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT_RMSNORM").is_err();
        let lora_decode_add_enabled = std::env::var("KILN_DISABLE_CUDA_LORA_DECODE_ADD").is_err();
        let gdn_full_chunk_forward_multiblock_enabled = gdn_enabled
            && std::env::var("KILN_DISABLE_GDN_FULL_CHUNK_FORWARD_MULTIBLOCK").is_err();
        Self {
            device,
            gdn_enabled,
            gdn_gates_enabled,
            gdn_gated_rms_norm_enabled,
            gdn_decode_fused_enabled,
            gdn_decode_unexpanded_qk_enabled,
            gdn_decode_qk_norm_recurrent_enabled,
            gdn_decode_qk_norm_recurrent_rmsnorm_enabled,
            fused_conv1d_enabled,
            cuda_use_kt_api_sgd_step,
            cuda_use_kt_api_adamw_step,
            lora_decode_add_enabled,
            gdn_full_chunk_forward_multiblock_enabled,
        }
    }

    pub fn training_capabilities_static() -> TrainingCapabilities {
        TrainingCapabilities {
            projection_training: "backend-routed candle CUDA autograd with offset chunk hook",
            flce_loss: "FLCE CustomOp on CUDA tensors; no full logits by default",
            rmsnorm_training: "CUDA CustomOp2 behind 47 GiB autograd VRAM gate",
            resident_activation: "TensorId lifecycle registry; candle CUDA tensors are canonical",
            lora_delta_training: "registered candle CUDA autograd; fused lora_decode_add declines tracked tensors",
            sgd_step: "CUDA in-place optimizer kernel for resident contiguous F32/BF16 tensors",
            adamw_step: "CUDA in-place optimizer kernel for resident contiguous F32/BF16 tensors",
            native_training: "not implemented",
        }
    }
}

impl BackendRuntime for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn training_capabilities(&self) -> TrainingCapabilities {
        Self::training_capabilities_static()
    }

    fn supports_resident_activation(&self) -> bool {
        true
    }

    fn register_resident_activation(&self, tensor: &Tensor) -> Result<()> {
        with_cuda_resident_ids(|ids| {
            ids.insert(tensor.id());
        });
        Ok(())
    }

    fn evict_resident_activation(&self, tensor: &Tensor) {
        with_cuda_resident_ids(|ids| {
            ids.remove(&tensor.id());
        });
    }

    fn update_resident_activation(&self, tensor: &Tensor) -> Result<()> {
        with_cuda_resident_ids(|ids| {
            ids.insert(tensor.id());
        });
        Ok(())
    }

    fn has_resident_activation(&self, tensor: &Tensor) -> bool {
        with_cuda_resident_ids(|ids| ids.contains(&tensor.id()))
    }

    fn dispatch_sgd_step(&self, param: &Tensor, grad: &Tensor, lr: f32) -> Result<bool> {
        if !self.has_resident_activation(param) || !self.has_resident_activation(grad) {
            return Ok(false);
        }
        if !kiln_rmsnorm_kernel::supports_optimizer_step(&[param, grad]) {
            return Ok(false);
        }
        // Phase 7 opt-in (#1082): route through the kt-typed surface.
        // Bit-exact by construction — both candle and kt paths bottom
        // out in the same `kiln_sgd_step_{f32,bf16}` FFI symbols; the
        // in-place mutation surfaces in the caller's candle tensor
        // because the kt borrow adapter is zero-copy.
        if self.cuda_use_kt_api_sgd_step {
            kiln_nvtx::range!(c"kiln/sgd_step_kt");
            let param_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(param)
                .context("sgd_step kt: borrow param -> kt")?;
            let grad_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(grad)
                .context("sgd_step kt: borrow grad -> kt")?;
            match param.dtype() {
                DType::F32 => kiln_rmsnorm_kernel::sgd_step_f32_kt(&param_kt, &grad_kt, lr)
                    .map_err(|e| anyhow::anyhow!("sgd_step kt: sgd_step_f32_kt: {e}"))?,
                DType::BF16 => kiln_rmsnorm_kernel::sgd_step_bf16_kt(&param_kt, &grad_kt, lr)
                    .map_err(|e| anyhow::anyhow!("sgd_step kt: sgd_step_bf16_kt: {e}"))?,
                other => anyhow::bail!("sgd_step kt: unsupported dtype {other:?}"),
            }
        } else {
            kiln_rmsnorm_kernel::sgd_step_inplace(param, grad, lr)
                .context("cuda dispatch_sgd_step kernel failed")?;
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

    #[allow(clippy::too_many_arguments)]
    fn dispatch_adamw_step(
        &self,
        param: &Tensor,
        grad: &Tensor,
        first_moment: &Tensor,
        second_moment: &Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> Result<bool> {
        if !self.has_resident_activation(param)
            || !self.has_resident_activation(grad)
            || !self.has_resident_activation(first_moment)
            || !self.has_resident_activation(second_moment)
        {
            return Ok(false);
        }
        if !kiln_rmsnorm_kernel::supports_optimizer_step(&[
            param,
            grad,
            first_moment,
            second_moment,
        ]) {
            return Ok(false);
        }
        // Phase 7 opt-in (#1082): route through the kt-typed surface.
        // Bit-exact by construction — both candle and kt paths bottom
        // out in the same `kiln_adamw_step_{f32,bf16}` FFI symbols;
        // the in-place mutation surfaces in the caller's candle
        // tensors because the kt borrow adapter is zero-copy. The
        // candle path computes the bias-correction terms internally
        // from `step: u32`; the kt path takes pre-computed
        // `bias_correction1` / `bias_correction2` instead, so we
        // replicate the same `(1 - beta^step).max(1e-20)` formula
        // (matches `kiln_rmsnorm_kernel::adamw_step_inplace`).
        if self.cuda_use_kt_api_adamw_step {
            if step == 0 {
                anyhow::bail!("adamw_step kt: step must be >= 1");
            }
            kiln_nvtx::range!(c"kiln/adamw_step_kt");
            let bias_correction1 = (1.0f32 - beta1.powi(step as i32)).max(1e-20);
            let bias_correction2 = (1.0f32 - beta2.powi(step as i32)).max(1e-20);
            let param_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(param)
                .context("adamw_step kt: borrow param -> kt")?;
            let grad_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(grad)
                .context("adamw_step kt: borrow grad -> kt")?;
            let m1_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(first_moment)
                .context("adamw_step kt: borrow first_moment -> kt")?;
            let m2_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(second_moment)
                .context("adamw_step kt: borrow second_moment -> kt")?;
            match param.dtype() {
                DType::F32 => kiln_rmsnorm_kernel::adamw_step_f32_kt(
                    &param_kt,
                    &grad_kt,
                    &m1_kt,
                    &m2_kt,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    bias_correction1,
                    bias_correction2,
                )
                .map_err(|e| anyhow::anyhow!("adamw_step kt: adamw_step_f32_kt: {e}"))?,
                DType::BF16 => kiln_rmsnorm_kernel::adamw_step_bf16_kt(
                    &param_kt,
                    &grad_kt,
                    &m1_kt,
                    &m2_kt,
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
        } else {
            kiln_rmsnorm_kernel::adamw_step_inplace(
                param,
                grad,
                first_moment,
                second_moment,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            )
            .context("cuda dispatch_adamw_step kernel failed")?;
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

    fn supports_flash_attn_prefill(&self) -> bool {
        true
    }

    fn supports_flash_attn_paged_decode(&self) -> bool {
        true
    }

    /// CUDA has no impl for the strict `flash_attn_paged_decode_contiguous_batch`
    /// kernel (the bs>1 head-major uniform-`start_pos` path), so the trait
    /// default `Ok(None)` always declines. Returning `false` here lets the
    /// `try_strict` probe in `gqa_attention_paged_decode_contiguous_batch`
    /// skip the `start_slots = Tensor::from_slice(...)` allocation that
    /// would otherwise emit a captured `cudaMemcpyHtoDAsync` to a recycled
    /// VA under CUDA graph capture (suspect 6 in
    /// `bench-results/cuda-graph-bs2-secondary-audit.md`, #1082).
    fn supports_strict_paged_decode_contiguous_batch(&self) -> bool {
        false
    }

    fn supports_gdn_forward_substitution(&self) -> bool {
        self.gdn_enabled
    }

    fn supports_gdn_recurrent_step(&self) -> bool {
        self.gdn_enabled
    }

    fn supports_gdn_chunk_prep(&self) -> bool {
        self.gdn_enabled
    }

    fn supports_gdn_chunk_scan(&self) -> bool {
        self.gdn_enabled
    }

    fn supports_gdn_full_chunk_forward(&self) -> bool {
        self.gdn_enabled
    }

    fn supports_gdn_decode_gates_recurrent_unexpanded_qk(&self) -> bool {
        self.gdn_decode_unexpanded_qk_enabled
    }

    fn supports_gdn_decode_qk_norm_gates_recurrent(&self) -> bool {
        self.gdn_decode_qk_norm_recurrent_enabled
    }

    fn flash_attn_prefill(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        // The vendored CUDA kernel hard-errors on non-BF16. Decline here so
        // the caller falls back to the portable path instead of bubbling a
        // hard error up for non-BF16 test configs.
        if any_tracks_op(&[q, k, v]) {
            CUDA_FLASH_ATTN_TRACKED_DECLINES.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        if q.dtype() != DType::BF16 {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed surface is now the only path.
        // Same closeout pattern as conv1d (2ebcfb08), marlin
        // (0841c266), GDN (86c7f134). Bit-exact: bottoms out in
        // the same `kiln_flash_attn_fwd` FFI symbol as the candle
        // shim; candle wrapper discards softmax_lse, kt path does
        // the same here.
        kiln_nvtx::range!(c"kiln/flash_attn_kt");
        let q_c = q.contiguous().context("flash_attn kt: q contiguous")?;
        let k_c = k.contiguous().context("flash_attn kt: k contiguous")?;
        let v_c = v.contiguous().context("flash_attn kt: v contiguous")?;
        let q_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&q_c)
            .context("flash_attn kt: borrow q -> kt")?;
        let k_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&k_c)
            .context("flash_attn kt: borrow k -> kt")?;
        let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&v_c)
            .context("flash_attn kt: borrow v -> kt")?;
        let (out_kt, _lse_kt) =
            kiln_flash_attn::flash_attn_fwd_kt(&q_kt, &k_kt, &v_kt, softmax_scale, causal)
                .map_err(|e| anyhow::anyhow!("flash_attn kt: flash_attn_fwd_kt: {e}"))?;
        let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
            .context("flash_attn kt: copy kt out -> candle")?;
        Ok(Some(out))
    }

    fn flash_attn_paged_decode(
        &self,
        q: &Tensor,
        k_pool: &Tensor,
        v_pool: &Tensor,
        block_table: &Tensor,
        total_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        if any_tracks_op(&[q, k_pool, v_pool, block_table]) || q.dtype() != DType::BF16 {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-only. Bit-exact with the candle path
        // — both bottom out in `kiln_flash_attn_fwd_paged_decode`.
        kiln_nvtx::range!(c"kiln/flash_attn_paged_decode_kt");
        let q_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(q)
            .context("flash_attn_paged_decode kt: borrow q -> kt")?;
        let k_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(k_pool)
            .context("flash_attn_paged_decode kt: borrow k_pool -> kt")?;
        let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(v_pool)
            .context("flash_attn_paged_decode kt: borrow v_pool -> kt")?;
        let bt_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(block_table)
            .context("flash_attn_paged_decode kt: borrow block_table -> kt")?;
        let out_kt = kiln_flash_attn::flash_attn_paged_decode_kt(
            &q_kt,
            &k_kt,
            &v_kt,
            &bt_kt,
            total_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
        .map_err(|e| anyhow::anyhow!("flash_attn_paged_decode kt: {e}"))?;
        let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
            .context("flash_attn_paged_decode kt: copy kt out -> candle")?;
        Ok(Some(out))
    }

    fn flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
        &self,
        q: &Tensor,
        k_pool: &Tensor,
        v_pool: &Tensor,
        block_table: &Tensor,
        seqused_k: &Tensor,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        if any_tracks_op(&[q, k_pool, v_pool, block_table, seqused_k]) || q.dtype() != DType::BF16 {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-only. Bit-exact with the candle path
        // — both bottom out in
        // `kiln_flash_attn_fwd_paged_decode_dyn_seqlen`. This entry
        // always passed `graph_outputs = None`; the caller-owned-
        // output variant lives in `_with_graph_outputs` below and
        // still uses the candle wrapper because the kt-typed entry
        // doesn't accept a caller-owned (out, lse) pair yet.
        kiln_nvtx::range!(c"kiln/flash_attn_paged_decode_dyn_seqlen_kt");
        let q_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(q)
            .context("flash_attn_paged_decode_dyn_seqlen kt: borrow q -> kt")?;
        let k_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(k_pool)
            .context("flash_attn_paged_decode_dyn_seqlen kt: borrow k_pool -> kt")?;
        let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(v_pool)
            .context("flash_attn_paged_decode_dyn_seqlen kt: borrow v_pool -> kt")?;
        let bt_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(block_table)
            .context("flash_attn_paged_decode_dyn_seqlen kt: borrow block_table -> kt")?;
        let sk_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(seqused_k)
            .context("flash_attn_paged_decode_dyn_seqlen kt: borrow seqused_k -> kt")?;
        let out_kt = kiln_flash_attn::flash_attn_paged_decode_dyn_seqlen_kt(
            &q_kt,
            &k_kt,
            &v_kt,
            &bt_kt,
            &sk_kt,
            max_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
        .map_err(|e| anyhow::anyhow!("flash_attn_paged_decode_dyn_seqlen kt: {e}"))?;
        let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
            .context("flash_attn_paged_decode_dyn_seqlen kt: copy kt out -> candle")?;
        Ok(Some(out))
    }

    fn flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs(
        &self,
        q: &Tensor,
        k_pool: &Tensor,
        v_pool: &Tensor,
        block_table: &Tensor,
        seqused_k: &Tensor,
        graph_outputs: Option<(&Tensor, &Tensor)>,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        if any_tracks_op(&[q, k_pool, v_pool, block_table, seqused_k]) || q.dtype() != DType::BF16 {
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
        // Phase 7 (#1082): kt path when graph_outputs is None;
        // candle path only when caller owns the (out, lse) pair
        // (the kt-typed entry doesn't accept caller-owned outputs
        // yet — that's the next migration). The flag gate was
        // removed alongside the same flag's other 3 sites; the
        // condition reduces to `graph_outputs.is_none()`.
        if graph_outputs.is_none() {
            kiln_nvtx::range!(c"kiln/flash_attn_paged_decode_dyn_seqlen_kt");
            let q_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(q)
                .context("flash_attn_paged_decode_dyn_seqlen (graph variant) kt: borrow q -> kt")?;
            let k_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(k_pool).context(
                "flash_attn_paged_decode_dyn_seqlen (graph variant) kt: borrow k_pool -> kt",
            )?;
            let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(v_pool).context(
                "flash_attn_paged_decode_dyn_seqlen (graph variant) kt: borrow v_pool -> kt",
            )?;
            let bt_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(block_table).context(
                "flash_attn_paged_decode_dyn_seqlen (graph variant) kt: borrow block_table -> kt",
            )?;
            let sk_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(seqused_k).context(
                "flash_attn_paged_decode_dyn_seqlen (graph variant) kt: borrow seqused_k -> kt",
            )?;
            let out_kt = kiln_flash_attn::flash_attn_paged_decode_dyn_seqlen_kt(
                &q_kt,
                &k_kt,
                &v_kt,
                &bt_kt,
                &sk_kt,
                max_seqlen_k,
                page_block_size,
                softmax_scale,
                causal,
            )
            .map_err(|e| {
                anyhow::anyhow!("flash_attn_paged_decode_dyn_seqlen (graph variant) kt: {e}")
            })?;
            let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt).context(
                "flash_attn_paged_decode_dyn_seqlen (graph variant) kt: copy kt out -> candle",
            )?;
            return Ok(Some(out));
        }
        // Pass `graph_outputs` straight through to the kernel wrapper. When
        // `Some`, the wrapper skips its `Tensor::zeros((b, 1, n_heads,
        // head_dim))` and `Tensor::zeros((b, n_heads, 1))` allocations and
        // writes directly into the caller-owned tensors — which the CUDA
        // graph runner re-uses across replays — fixing the
        // dangling-pointer hazard documented in
        // `bench-results/cuda-graph-bs2-secondary-audit.md` suspects 3+4
        // (#1082).
        let out = kiln_flash_attn::flash_attn_paged_decode_dyn_seqlen(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            graph_outputs,
            max_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
        .context("flash_attn_paged_decode_dyn_seqlen kernel failed (graph outputs)")?;
        Ok(Some(out))
    }

    fn gdn_forward_substitution(
        &self,
        a_strict: &Tensor,
        v_prime: &Tensor,
        beta: &Tensor,
    ) -> Result<Option<Tensor>> {
        if a_strict.dtype() != DType::BF16 {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed surface is now the only path,
        // same closeout pattern as conv1d (2ebcfb08) and marlin
        // (0841c266). The borrow adapter shares the underlying CUDA
        // buffer with the candle tensor, so this is zero-copy on
        // inputs. The output is copied back via
        // `kt_tensor_to_candle_cuda_copy` (one dtod memcpy on the F32
        // forward-substitution result, mirrors marlin pattern).
        kiln_nvtx::range!(c"kiln/gdn_forward_substitution_kt");
        let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(a_strict)
            .with_context(|| "kt-adapter: gdn_forward_substitution a_strict → kt failed")?;
        let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(v_prime)
            .with_context(|| "kt-adapter: gdn_forward_substitution v_prime → kt failed")?;
        let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(beta)
            .with_context(|| "kt-adapter: gdn_forward_substitution beta → kt failed")?;
        let out_kt = kiln_gdn_kernel::gdn_forward_substitution_kt(&a_kt, &v_kt, &b_kt)
            .map_err(|e| anyhow::anyhow!("kt gdn_forward_substitution: {e}"))?;
        let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
            .with_context(|| "kt-adapter: gdn_forward_substitution out → candle failed")?;
        Ok(Some(out))
    }

    fn gdn_recurrent_step(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        if q.dtype() != DType::BF16 {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed surface is now the only path,
        // same closeout pattern as conv1d (2ebcfb08) and marlin
        // (0841c266). The borrow adapter shares the underlying CUDA
        // buffer with the candle tensor, so the kernel's in-place
        // mutation of `state` surfaces through the caller's
        // `&mut Tensor` (same pattern as conv1d_update at 695587df).
        kiln_nvtx::range!(c"kiln/gdn_recurrent_forward_kt");
        let q_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(q)
            .with_context(|| "kt-adapter: gdn_recurrent_step q → kt failed")?;
        let k_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(k)
            .with_context(|| "kt-adapter: gdn_recurrent_step k → kt failed")?;
        let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(v)
            .with_context(|| "kt-adapter: gdn_recurrent_step v → kt failed")?;
        let beta_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(beta)
            .with_context(|| "kt-adapter: gdn_recurrent_step beta → kt failed")?;
        let g_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(g)
            .with_context(|| "kt-adapter: gdn_recurrent_step g → kt failed")?;
        let state_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(state)
            .with_context(|| "kt-adapter: gdn_recurrent_step state → kt failed")?;
        let out_kt = kiln_gdn_kernel::gdn_recurrent_forward_kt(
            &q_kt, &k_kt, &v_kt, &beta_kt, &g_kt, &state_kt,
        )
        .map_err(|e| anyhow::anyhow!("kt gdn_recurrent_forward: {e}"))?;
        let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
            .with_context(|| "kt-adapter: gdn_recurrent_step out → candle failed")?;
        Ok(Some(out))
    }

    fn gdn_chunk_prep(
        &self,
        g: &Tensor,
        v: &Tensor,
        kkt: &Tensor,
        qkt: &Tensor,
        ks_entry: &Tensor,
        q_s: &Tensor,
    ) -> Result<Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)>> {
        // Phase 7 (#1082): kt-typed surface is now the only path,
        // same closeout pattern as conv1d (2ebcfb08) and marlin
        // (0841c266). The kt-typed predicate (`_supports_kt`,
        // 7da2615a) and the `gdn_chunk_prep_kt` 6-tuple-returning
        // kernel are the only call path — the candle-typed
        // `kiln_gdn_kernel::gdn_chunk_prep[_supports]` fallback has
        // been deleted.
        kiln_nvtx::range!(c"kiln/gdn_chunk_prep_kt");
        let g_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(g)
            .with_context(|| "kt-adapter: gdn_chunk_prep g → kt failed")?;
        let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(v)
            .with_context(|| "kt-adapter: gdn_chunk_prep v → kt failed")?;
        let kkt_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(kkt)
            .with_context(|| "kt-adapter: gdn_chunk_prep kkt → kt failed")?;
        let qkt_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(qkt)
            .with_context(|| "kt-adapter: gdn_chunk_prep qkt → kt failed")?;
        let ks_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(ks_entry)
            .with_context(|| "kt-adapter: gdn_chunk_prep ks_entry → kt failed")?;
        let qs_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(q_s)
            .with_context(|| "kt-adapter: gdn_chunk_prep q_s → kt failed")?;
        if !kiln_gdn_kernel::gdn_chunk_prep_supports_kt(
            &g_kt, &v_kt, &kkt_kt, &qkt_kt, &ks_kt, &qs_kt,
        ) {
            return Ok(None);
        }
        let (o0, o1, o2, o3, o4, o5) =
            kiln_gdn_kernel::gdn_chunk_prep_kt(&g_kt, &v_kt, &kkt_kt, &qkt_kt, &ks_kt, &qs_kt)
                .map_err(|e| anyhow::anyhow!("kt gdn_chunk_prep: {e}"))?;
        let c0 = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&o0)
            .with_context(|| "kt-adapter: gdn_chunk_prep o0 → candle failed")?;
        let c1 = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&o1)
            .with_context(|| "kt-adapter: gdn_chunk_prep o1 → candle failed")?;
        let c2 = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&o2)
            .with_context(|| "kt-adapter: gdn_chunk_prep o2 → candle failed")?;
        let c3 = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&o3)
            .with_context(|| "kt-adapter: gdn_chunk_prep o3 → candle failed")?;
        let c4 = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&o4)
            .with_context(|| "kt-adapter: gdn_chunk_prep o4 → candle failed")?;
        let c5 = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&o5)
            .with_context(|| "kt-adapter: gdn_chunk_prep o5 → candle failed")?;
        Ok(Some((c0, c1, c2, c3, c4, c5)))
    }

    fn gdn_chunk_scan(
        &self,
        a_strict: &Tensor,
        b_mask: &Tensor,
        v_prime: &Tensor,
        q_s_scaled: &Tensor,
        beta: &Tensor,
        decay_last_col: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        // Phase 7 (#1082): kt-typed surface is now the only path,
        // same closeout pattern as conv1d (2ebcfb08) and marlin
        // (0841c266). The kt-typed predicate (`_supports_kt`,
        // 7da2615a) and the `gdn_chunk_scan_kt` 2-tuple-returning
        // kernel are the only call path — the candle-typed
        // `kiln_gdn_kernel::gdn_chunk_scan[_supports]` fallback has
        // been deleted.
        kiln_nvtx::range!(c"kiln/gdn_chunk_scan_kt");
        let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(a_strict)
            .with_context(|| "kt-adapter: gdn_chunk_scan a_strict → kt failed")?;
        let m_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(b_mask)
            .with_context(|| "kt-adapter: gdn_chunk_scan b_mask → kt failed")?;
        let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(v_prime)
            .with_context(|| "kt-adapter: gdn_chunk_scan v_prime → kt failed")?;
        let qs_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(q_s_scaled)
            .with_context(|| "kt-adapter: gdn_chunk_scan q_s_scaled → kt failed")?;
        let beta_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(beta)
            .with_context(|| "kt-adapter: gdn_chunk_scan beta → kt failed")?;
        let dlc_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(decay_last_col)
            .with_context(|| "kt-adapter: gdn_chunk_scan decay_last_col → kt failed")?;
        if !kiln_gdn_kernel::gdn_chunk_scan_supports_kt(
            &a_kt, &m_kt, &v_kt, &qs_kt, &beta_kt, &dlc_kt,
        ) {
            return Ok(None);
        }
        let (o0, o1) = kiln_gdn_kernel::gdn_chunk_scan_kt(
            &a_kt, &m_kt, &v_kt, &qs_kt, &beta_kt, &dlc_kt,
        )
        .map_err(|e| anyhow::anyhow!("kt gdn_chunk_scan: {e}"))?;
        let c0 = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&o0)
            .with_context(|| "kt-adapter: gdn_chunk_scan o0 → candle failed")?;
        let c1 = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&o1)
            .with_context(|| "kt-adapter: gdn_chunk_scan o1 → candle failed")?;
        Ok(Some((c0, c1)))
    }

    fn gdn_full_chunk_forward(
        &self,
        g: &Tensor,
        v: &Tensor,
        kkt: &Tensor,
        qkt: &Tensor,
        ks_entry: &Tensor,
        q_s: &Tensor,
        beta: &Tensor,
        k_t: &Tensor,
        state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        let dv_tile = kiln_gdn_kernel::GDN_FULL_CHUNK_FORWARD_MULTIBLOCK_DV_TILE;
        // Phase 7 (#1082): kt-typed surface is now the only path,
        // same closeout pattern as conv1d (2ebcfb08) and marlin
        // (0841c266). Both kt-typed predicates (`_supports_kt`,
        // `_multiblock_supports_kt`, 7da2615a) and the
        // multiblock_kt kernel are the only candle-free wires. The
        // single-block fall-through still uses the candle-typed
        // `kiln_gdn_kernel::gdn_full_chunk_forward` because the
        // kt-typed single-block kernel wire does not exist yet —
        // see the comment below. State mutation surfaces through
        // the caller's `&mut Tensor` via the shared buffer (conv1d
        // pattern).
        let g_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(g)
            .with_context(|| "kt-adapter: gdn_full_chunk_forward g → kt")?;
        let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(v)
            .with_context(|| "kt-adapter: gdn_full_chunk_forward v → kt")?;
        let kkt_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(kkt)
            .with_context(|| "kt-adapter: gdn_full_chunk_forward kkt → kt")?;
        let qkt_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(qkt)
            .with_context(|| "kt-adapter: gdn_full_chunk_forward qkt → kt")?;
        let ks_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(ks_entry)
            .with_context(|| "kt-adapter: gdn_full_chunk_forward ks_entry → kt")?;
        let qs_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(q_s)
            .with_context(|| "kt-adapter: gdn_full_chunk_forward q_s → kt")?;
        let beta_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(beta)
            .with_context(|| "kt-adapter: gdn_full_chunk_forward beta → kt")?;
        let kt_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(k_t)
            .with_context(|| "kt-adapter: gdn_full_chunk_forward k_t → kt")?;
        let state_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(state)
            .with_context(|| "kt-adapter: gdn_full_chunk_forward state → kt")?;
        if !kiln_gdn_kernel::gdn_full_chunk_forward_supports_kt(
            &g_kt, &v_kt, &kkt_kt, &qkt_kt, &ks_kt, &qs_kt, &beta_kt, &kt_kt, &state_kt,
        ) {
            return Ok(None);
        }
        if self.gdn_full_chunk_forward_multiblock_enabled
            && kiln_gdn_kernel::gdn_full_chunk_forward_multiblock_supports_kt(
                &g_kt, &v_kt, &kkt_kt, &qkt_kt, &ks_kt, &qs_kt, &beta_kt, &kt_kt, &state_kt,
                dv_tile,
            )
        {
            kiln_nvtx::range!(c"kiln/gdn_full_chunk_forward_multiblock_kt");
            let out_kt = kiln_gdn_kernel::gdn_full_chunk_forward_multiblock_kt(
                &g_kt, &v_kt, &kkt_kt, &qkt_kt, &ks_kt, &qs_kt, &beta_kt, &kt_kt, &state_kt,
                dv_tile,
            )
            .map_err(|e| anyhow::anyhow!("kt gdn_full_chunk_forward_multiblock: {e}"))?;
            let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
                .with_context(|| "kt-adapter: gdn_full_chunk_forward_multiblock out → candle")?;
            CUDA_GDN_FULL_CHUNK_FORWARD_MULTIBLOCK_SUCCESSES.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(out));
        }
        // Single-block fall-through. We've borrowed all tensors as kt
        // and confirmed kt-typed `_supports_kt` passes, but there is
        // no kt-typed single-block kernel wire yet — fall back to the
        // candle dispatch using the original candle `&Tensor` refs we
        // still hold (the kt borrows are zero-copy adapters over the
        // same underlying device storage, so the candle refs remain
        // valid). Once a `gdn_full_chunk_forward_kt` lands, this
        // candle call should be the next thing to migrate.
        let out = kiln_gdn_kernel::gdn_full_chunk_forward(
            g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state,
        )
        .context("gdn_full_chunk_forward kernel failed")?;
        CUDA_GDN_FULL_CHUNK_FORWARD_SINGLE_SUCCESSES.fetch_add(1, Ordering::Relaxed);
        Ok(Some(out))
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_gates_recurrent(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        state: &mut Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f64,
    ) -> Result<Option<Tensor>> {
        if !self.gdn_decode_fused_enabled {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed bf16 surface is now the only path,
        // same closeout pattern as conv1d (2ebcfb08) and marlin
        // (0841c266). The candle-typed
        // `kiln_gdn_kernel::gdn_decode_gates_recurrent[_supports]`
        // fallback has been deleted; non-bf16 envelopes return
        // Ok(None) so the caller's split-decode fallback engages.
        // Production decode for Qwen3.5-4B uses bf16 for q/k/v/a/b/
        // a_log/dt_bias/state/z and f32 for the rmsnorm weight; the
        // bf16_kt variant is the matching production hot path.
        if !(q.dtype() == DType::BF16
            && k.dtype() == DType::BF16
            && v.dtype() == DType::BF16
            && a.dtype() == DType::BF16
            && b.dtype() == DType::BF16
            && a_log.dtype() == DType::BF16
            && dt_bias.dtype() == DType::BF16
            && state.dtype() == DType::BF16
            && z.dtype() == DType::BF16
            && weight.dtype() == DType::F32)
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
                "CUDA gdn_decode_gates_recurrent declined (non-bf16 envelope); using split decode path"
            );
            return Ok(None);
        }
        kiln_nvtx::range!(c"kiln/gdn_decode_gates_recurrent_bf16_kt");
        // kt_api expects 3D [B, heads, dim] but the candle method
        // receives 4D [B, 1, heads, dim]. Squeeze the seq_len=1
        // axis (metadata-only reshape — no copy). Without this the
        // kt path errors at the very first shape check and the
        // gate is effectively dead in production. Same latent bug
        // as the rmsnorm wire fixed in 171020c.
        let q_3d = q.squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_gates q squeeze(1) failed")?;
        let k_3d = k.squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_gates k squeeze(1) failed")?;
        let v_3d = v.squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_gates v squeeze(1) failed")?;
        let q_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&q_3d)
            .with_context(|| "kt-adapter: gdn_decode_gates q → kt failed")?;
        let k_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&k_3d)
            .with_context(|| "kt-adapter: gdn_decode_gates k → kt failed")?;
        let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&v_3d)
            .with_context(|| "kt-adapter: gdn_decode_gates v → kt failed")?;
        let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(a)
            .with_context(|| "kt-adapter: gdn_decode_gates a → kt failed")?;
        let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(b)
            .with_context(|| "kt-adapter: gdn_decode_gates b → kt failed")?;
        let alog_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(a_log)
            .with_context(|| "kt-adapter: gdn_decode_gates a_log → kt failed")?;
        let dtb_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(dt_bias)
            .with_context(|| "kt-adapter: gdn_decode_gates dt_bias → kt failed")?;
        let state_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(state)
            .with_context(|| "kt-adapter: gdn_decode_gates state → kt failed")?;
        let z_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(z)
            .with_context(|| "kt-adapter: gdn_decode_gates z → kt failed")?;
        let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(weight)
            .with_context(|| "kt-adapter: gdn_decode_gates weight → kt failed")?;
        if !kiln_gdn_kernel::gdn_decode_gates_recurrent_supports_kt(
            &q_kt, &k_kt, &v_kt, &a_kt, &b_kt, &alog_kt, &dtb_kt, &state_kt, &z_kt, &w_kt,
        ) {
            return Ok(None);
        }
        let out_kt = kiln_gdn_kernel::gdn_decode_gates_recurrent_bf16_kt(
            &q_kt, &k_kt, &v_kt, &a_kt, &b_kt, &alog_kt, &dtb_kt, &state_kt, &z_kt, &w_kt,
            eps as f32,
        )
        .map_err(|e| anyhow::anyhow!("kt gdn_decode_gates_recurrent: {e}"))?;
        let out_3d = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
            .with_context(|| "kt-adapter: gdn_decode_gates out → candle failed")?;
        // kt_api allocates a 3D `[B, value_heads, dv]` output (see
        // crates/kiln-gdn-kernel/src/kt_api.rs:568) but the
        // BackendRuntime trait + production caller expect 4D
        // `[B, 1, value_heads, dv]` (see attn_out shape contract at
        // forward.rs:15595). Unsqueeze the seq_len axis back at
        // position 1; metadata-only reshape, no copy. (#1082)
        let out = out_3d
            .unsqueeze(1)
            .with_context(|| "kt-adapter: gdn_decode_gates out 3D->4D unsqueeze failed")?;
        Ok(Some(out))
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_qk_norm_gates_recurrent(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        state: &mut Tensor,
        q_scale: f64,
        qk_eps: f64,
    ) -> Result<Option<Tensor>> {
        if !self.gdn_decode_qk_norm_recurrent_enabled {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed bf16 surface is now the only path,
        // same closeout pattern as conv1d (2ebcfb08) and marlin
        // (0841c266). The candle-typed
        // `kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent
        // [_supports]` fallback has been deleted; non-bf16 envelopes
        // return Ok(None) so the caller's split qk_norm fallback
        // engages. Production decode for Qwen3.5-4B uses bf16 for all
        // 8 input tensors; the bf16_kt variant is the matching
        // production hot path.
        if !(q.dtype() == DType::BF16
            && k.dtype() == DType::BF16
            && v.dtype() == DType::BF16
            && a.dtype() == DType::BF16
            && b.dtype() == DType::BF16
            && a_log.dtype() == DType::BF16
            && dt_bias.dtype() == DType::BF16
            && state.dtype() == DType::BF16)
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
                "CUDA gdn_decode_qk_norm_gates_recurrent declined (non-bf16 envelope); using split qk_norm path"
            );
            return Ok(None);
        }
        kiln_nvtx::range!(c"kiln/gdn_decode_qk_norm_gates_recurrent_bf16_kt");
        // kt_api expects 3D [B, heads, dim] but the candle method
        // receives 4D [B, 1, heads, dim]. Squeeze the seq_len=1
        // axis (metadata-only reshape — no copy). Without this the
        // kt path errors at the very first shape check and the
        // gate is effectively dead in production. Same latent bug
        // as the rmsnorm wire fixed in 171020c.
        let q_3d = q.squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm q squeeze(1) failed")?;
        let k_3d = k.squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm k squeeze(1) failed")?;
        let v_3d = v.squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm v squeeze(1) failed")?;
        let q_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&q_3d)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm q → kt failed")?;
        let k_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&k_3d)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm k → kt failed")?;
        let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&v_3d)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm v → kt failed")?;
        let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(a)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm a → kt failed")?;
        let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(b)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm b → kt failed")?;
        let alog_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(a_log)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm a_log → kt failed")?;
        let dtb_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(dt_bias)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm dt_bias → kt failed")?;
        let state_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(state)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm state → kt failed")?;
        if !kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent_supports_kt(
            &q_kt, &k_kt, &v_kt, &a_kt, &b_kt, &alog_kt, &dtb_kt, &state_kt,
        ) {
            return Ok(None);
        }
        let out_kt = kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent_bf16_kt(
            &q_kt, &k_kt, &v_kt, &a_kt, &b_kt, &alog_kt, &dtb_kt, &state_kt,
            q_scale as f32, qk_eps as f32,
        )
        .map_err(|e| anyhow::anyhow!("kt gdn_decode_qk_norm_gates_recurrent: {e}"))?;
        let out_3d = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm out → candle failed")?;
        // Same 3D->4D unsqueeze fix as gdn_decode_gates_recurrent
        // above. The kt_api allocates 3D `[B, value_heads, dv]`
        // (see crates/kiln-gdn-kernel/src/kt_api.rs) but the trait
        // contract is 4D `[B, 1, value_heads, dv]`. (#1082)
        let out = out_3d
            .unsqueeze(1)
            .with_context(|| "kt-adapter: gdn_decode_qk_norm out 3D->4D unsqueeze failed")?;
        Ok(Some(out))
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_qk_norm_gates_recurrent_rmsnorm(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        state: &mut Tensor,
        z: &Tensor,
        weight: &Tensor,
        q_scale: f64,
        qk_eps: f64,
        rms_eps: f64,
    ) -> Result<Option<Tensor>> {
        if !self.gdn_decode_qk_norm_recurrent_rmsnorm_enabled {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed bf16 surface is now the only path,
        // same closeout pattern as conv1d (2ebcfb08) and marlin
        // (0841c266). The candle-typed
        // `kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent_rmsnorm
        // [_supports]` fallback has been deleted; non-bf16 envelopes
        // return Ok(None) so the caller's split gated_norm fallback
        // engages. Production decode for Qwen3.5-4B uses bf16 for all
        // 10 input tensors (with F32 weight); the bf16_kt variant is
        // the matching production hot path.
        if !(q.dtype() == DType::BF16
            && k.dtype() == DType::BF16
            && v.dtype() == DType::BF16
            && a.dtype() == DType::BF16
            && b.dtype() == DType::BF16
            && a_log.dtype() == DType::BF16
            && dt_bias.dtype() == DType::BF16
            && state.dtype() == DType::BF16
            && z.dtype() == DType::BF16
            && weight.dtype() == DType::F32)
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
                "CUDA gdn_decode_qk_norm_gates_recurrent_rmsnorm declined (non-bf16 envelope); using split gated_norm path"
            );
            return Ok(None);
        }
        kiln_nvtx::range!(c"kiln/gdn_decode_qk_norm_gates_recurrent_rmsnorm_bf16_kt");
        // kt_api expects 3D [B, heads, dim] but the candle method
        // receives 4D [B, 1, heads, dim]. Squeeze the seq_len=1
        // axis (metadata-only reshape — no copy). Without this the
        // kt path errors at the very first shape check and the
        // gate is effectively dead in production.
        let q_3d = q.squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm q squeeze(1) failed")?;
        let k_3d = k.squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm k squeeze(1) failed")?;
        let v_3d = v.squeeze(1)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm v squeeze(1) failed")?;
        let q_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&q_3d)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm q → kt failed")?;
        let k_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&k_3d)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm k → kt failed")?;
        let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&v_3d)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm v → kt failed")?;
        let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(a)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm a → kt failed")?;
        let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(b)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm b → kt failed")?;
        let alog_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(a_log)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm a_log → kt failed")?;
        let dtb_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(dt_bias)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm dt_bias → kt failed")?;
        let state_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(state)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm state → kt failed")?;
        let z_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(z)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm z → kt failed")?;
        let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(weight)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm weight → kt failed")?;
        if !kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent_rmsnorm_supports_kt(
            &q_kt, &k_kt, &v_kt, &a_kt, &b_kt, &alog_kt, &dtb_kt, &state_kt, &z_kt, &w_kt,
        ) {
            return Ok(None);
        }
        let out_kt = kiln_gdn_kernel::gdn_decode_qk_norm_gates_recurrent_rmsnorm_bf16_kt(
            &q_kt, &k_kt, &v_kt, &a_kt, &b_kt, &alog_kt, &dtb_kt, &state_kt, &z_kt, &w_kt,
            q_scale as f32, qk_eps as f32, rms_eps as f32,
        )
        .map_err(|e| anyhow::anyhow!("kt gdn_decode_qk_norm_gates_recurrent_rmsnorm: {e}"))?;
        let out_3d = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm out → candle failed")?;
        // Same 3D->4D unsqueeze fix as the gdn_decode_gates_recurrent
        // and gdn_decode_qk_norm_gates_recurrent wires above. The
        // kt_api allocates 3D `[B, value_heads, dv]` but the trait
        // contract is 4D `[B, 1, value_heads, dv]`. (#1082)
        let out = out_3d
            .unsqueeze(1)
            .with_context(|| "kt-adapter: gdn_decode_rmsnorm out 3D->4D unsqueeze failed")?;
        Ok(Some(out))
    }

    fn supports_gdn_gates(&self) -> bool {
        self.gdn_gates_enabled
    }

    fn gdn_gates(
        &self,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let dims = a.dims();
        let is_t1_decode = dims.len() >= 2 && dims[dims.len() - 2] == 1;
        if !is_t1_decode && std::env::var("KILN_DISABLE_CUDA_GDN_PREFILL_GATES").is_ok() {
            tracing::debug!(
                a_shape = ?a.shape(),
                a_log_dtype = ?a_log.dtype(),
                dt_bias_dtype = ?dt_bias.dtype(),
                "CUDA prefill gdn_gates disabled; using Candle fallback"
            );
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed bf16 surface is now the only path,
        // same closeout pattern as conv1d (2ebcfb08) and marlin
        // (0841c266). The candle-typed
        // `kiln_gdn_kernel::gdn_gates[_decline_reason]` fallback has
        // been deleted; non-bf16 envelopes (f32/f32, f32/bf16) return
        // Ok(None) so the caller's candle fallback engages. bf16 was
        // the only envelope on the production decode/prefill path for
        // Qwen3.5-4B GDN; the mixed-precision variants are reachable
        // only through paths not exercised in production.
        if !(a.dtype() == DType::BF16
            && b.dtype() == DType::BF16
            && a_log.dtype() == DType::BF16
            && dt_bias.dtype() == DType::BF16)
        {
            return Ok(None);
        }
        kiln_nvtx::range!(c"kiln/gdn_gates_bf16_kt");
        // kt_tensor_from_candle_cuda_borrow requires contiguous inputs (see
        // kiln-kt-bridge::lib.rs: "tensor must be contiguous"). At the
        // bs>1 / prefill GDN call site, `a` and `b` arrive as
        // `ab.narrow(2, .., nv)` views on a fused A/B in-proj output,
        // which are non-contiguous on the last dim. The candle path
        // used to handle this by computing a collapsed row-stride and
        // falling through to `.contiguous()` on declined stride; with
        // the candle path now gone, we make each operand contiguous
        // here unconditionally. This is a no-op when the upstream
        // tensor is already contiguous (the seq_len==1 decode case).
        // a_log and dt_bias are weight tensors and are already
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
        let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_c)
            .with_context(|| "kt-adapter: gdn_gates a → kt failed")?;
        let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_c)
            .with_context(|| "kt-adapter: gdn_gates b → kt failed")?;
        let alog_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&alog_c)
            .with_context(|| "kt-adapter: gdn_gates a_log → kt failed")?;
        let dtb_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&dtb_c)
            .with_context(|| "kt-adapter: gdn_gates dt_bias → kt failed")?;
        if !kiln_gdn_kernel::gdn_gates_supports_kt(&a_kt, &b_kt, &alog_kt, &dtb_kt) {
            return Ok(None);
        }
        let (beta_kt, g_kt) =
            kiln_gdn_kernel::gdn_gates_bf16_kt(&a_kt, &b_kt, &alog_kt, &dtb_kt)
                .map_err(|e| anyhow::anyhow!("kt gdn_gates_bf16: {e}"))?;
        let beta = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&beta_kt)
            .with_context(|| "kt-adapter: gdn_gates beta → candle failed")?;
        let g = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&g_kt)
            .with_context(|| "kt-adapter: gdn_gates g → candle failed")?;
        Ok(Some((beta, g)))
    }

    fn supports_gdn_gated_rms_norm(&self) -> bool {
        self.gdn_gated_rms_norm_enabled
    }

    fn lora_decode_add(
        &self,
        base: &Tensor,
        x: &Tensor,
        a: &Tensor,
        b: &Tensor,
        scale: f32,
    ) -> Result<Option<Tensor>> {
        if !self.lora_decode_add_enabled
            || base.track_op()
            || x.track_op()
            || a.track_op()
            || b.track_op()
            || !kiln_rmsnorm_kernel::supports_lora_decode_add(base, x, a, b)
        {
            return Ok(None);
        }
        let out = kiln_rmsnorm_kernel::lora_decode_add(base, x, a, b, scale)
            .context("cuda lora_decode_add kernel failed")?;
        Ok(Some(out))
    }

    fn linear_prefill_apply(&self, x: &Tensor, weight_t: &Tensor) -> Result<Option<Tensor>> {
        if !matches!(x.device(), Device::Cuda(_))
            || !matches!(weight_t.device(), Device::Cuda(_))
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
                "CudaBackend::linear_prefill_apply first call (candle CUDA autograd)"
            );
        });

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
            // when KILN_USE_KT_API_MATMUL=1 and the dtype is supported.
            // Brings the kt-API into the production CUDA prefill matmul
            // path (the training/autograd path that goes through
            // BackendRuntime::linear_prefill_apply before falling back
            // to matmul_no_broadcast_copy). NVTX range from try_kt_matmul
            // brackets the call as kiln/matmul_kt in nsys.
            if crate::forward::cuda_use_kt_api_matmul()
                && matches!(x2d.dtype(), candle_core::DType::BF16 | candle_core::DType::F16 | candle_core::DType::F32)
                && x2d.dtype() == weight_t.dtype()
                && x2d.is_contiguous()
                && weight_t.is_contiguous()
            {
                if let Some(kt_out2d) = crate::forward::try_kt_matmul(&x2d, weight_t)? {
                    let mut out_shape = l_dims[..l_dims.len() - 1].to_vec();
                    out_shape.push(out_n);
                    CUDA_LINEAR_PREFILL_SUCCESSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(Some(kt_out2d.reshape(out_shape)?));
                }
            }

            let out2d = x2d.matmul(weight_t)?;
            let mut out_shape = l_dims[..l_dims.len() - 1].to_vec();
            out_shape.push(out_n);
            out2d.reshape(out_shape)?
        } else {
            x.broadcast_matmul(weight_t)?
        };
        CUDA_LINEAR_PREFILL_SUCCESSES.fetch_add(1, Ordering::Relaxed);
        Ok(Some(out))
    }

    fn linear_prefill_apply_offset(
        &self,
        x: &Tensor,
        full_weight_t: &Tensor,
        chunk_start: usize,
        chunk_len: usize,
    ) -> Result<Option<Tensor>> {
        if !matches!(x.device(), Device::Cuda(_))
            || !matches!(full_weight_t.device(), Device::Cuda(_))
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
        let out = self.linear_prefill_apply(x, &chunk)?;
        if out.is_some() {
            CUDA_LINEAR_PREFILL_OFFSET_SUCCESSES.fetch_add(1, Ordering::Relaxed);
        }
        Ok(out)
    }

    fn lora_delta_resident(
        &self,
        x: &Tensor,
        a: &Tensor,
        b: &Tensor,
        scale: f32,
    ) -> Result<Option<Tensor>> {
        if !matches!(x.device(), Device::Cuda(_))
            || !matches!(a.device(), Device::Cuda(_))
            || !matches!(b.device(), Device::Cuda(_))
            || !self.has_resident_activation(a)
            || !self.has_resident_activation(b)
        {
            return Ok(None);
        }

        let proj = LoraProjectionWeights {
            a: a.clone(),
            b: b.clone(),
        };
        let delta = compute_lora_delta(x, &proj, scale)
            .context("cuda registered LoRA delta via candle CUDA autograd failed")?;

        static FIRST_CUDA_LORA_DELTA_LOGGED: OnceLock<()> = OnceLock::new();
        FIRST_CUDA_LORA_DELTA_LOGGED.get_or_init(|| {
            tracing::info!(
                x_shape = ?x.dims(),
                a_shape = ?a.dims(),
                b_shape = ?b.dims(),
                scale,
                "CudaBackend::lora_delta_resident first call (candle CUDA autograd)"
            );
        });

        Ok(Some(delta))
    }

    fn gdn_gated_rms_norm(
        &self,
        x: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f64,
    ) -> Result<Option<Tensor>> {
        if !self.gdn_gated_rms_norm_enabled {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed bf16 surface is now the only path,
        // same closeout pattern as conv1d (2ebcfb08) and marlin
        // (0841c266). The candle-typed `kiln_gdn_kernel::gdn_gated_rms_
        // norm[_supports]` fallback has been deleted; non-bf16 inputs
        // (which the kt path declines) return Ok(None) so the caller's
        // candle fallback engages. bf16 was the only envelope the
        // candle path could reach in production for Qwen3.5-4B GDN.
        if !(x.dtype() == DType::BF16
            && z.dtype() == DType::BF16
            && weight.dtype() == DType::BF16)
        {
            return Ok(None);
        }
        kiln_nvtx::range!(c"kiln/gdn_gated_rms_norm_bf16_kt");
        // The kt variant expects rank-2 [rows, hidden]; flatten higher-rank
        // x/z by folding all leading dims into rows. weight stays [hidden].
        let x_dims = x.dims();
        let hidden = *x_dims.last().expect("x has at least one dim (checked by supports_kt)");
        let rows: usize = x_dims.iter().take(x_dims.len() - 1).product();
        let x_flat = x
            .reshape((rows, hidden))
            .context("kt-adapter: gdn_gated_rms_norm reshape x → [rows, hidden] failed")?;
        let z_flat = z
            .reshape((rows, hidden))
            .context("kt-adapter: gdn_gated_rms_norm reshape z → [rows, hidden] failed")?;
        let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_flat)
            .with_context(|| "kt-adapter: gdn_gated_rms_norm x → kt failed")?;
        let z_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&z_flat)
            .with_context(|| "kt-adapter: gdn_gated_rms_norm z → kt failed")?;
        let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(weight)
            .with_context(|| "kt-adapter: gdn_gated_rms_norm weight → kt failed")?;
        if !kiln_gdn_kernel::gdn_gated_rms_norm_supports_kt(&x_kt, &z_kt, &w_kt) {
            return Ok(None);
        }
        let out_kt =
            kiln_gdn_kernel::gdn_gated_rms_norm_bf16_kt(&x_kt, &z_kt, &w_kt, eps as f32)
                .map_err(|e| anyhow::anyhow!("kt gdn_gated_rms_norm_bf16: {e}"))?;
        let out_flat = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
            .with_context(|| "kt-adapter: gdn_gated_rms_norm out → candle failed")?;
        let out = out_flat
            .reshape(x_dims)
            .context("kt-adapter: gdn_gated_rms_norm reshape out → original failed")?;
        Ok(Some(out))
    }

    fn supports_causal_conv1d_update(&self) -> bool {
        self.fused_conv1d_enabled
    }

    fn supports_causal_conv1d_prefill(&self) -> bool {
        self.fused_conv1d_enabled
    }

    fn causal_conv1d_update(
        &self,
        x: &Tensor,
        weight: &Tensor,
        conv_state: &mut Tensor,
        kernel_size: usize,
    ) -> Result<Option<Tensor>> {
        if !self.fused_conv1d_enabled {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed surface is now the only path.
        // The borrow adapter shares the underlying CUDA buffer with
        // the candle tensor, so the kernel's in-place mutation of
        // `conv_state` surfaces through the caller's `&mut Tensor`
        // automatically (anti-pattern 16 — owner-agnostic raw ptr).
        // Predicate also runs on the kt-borrowed view so no
        // candle-typed `kiln_conv1d_kernel::supports*` call survives
        // in production code — see docs/CANDLE_REMOVAL_PLAN.md
        // §"kiln-conv1d-kernel".
        let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(x)
            .with_context(|| "kt-adapter: conv1d_update x → kt failed")?;
        let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(weight)
            .with_context(|| "kt-adapter: conv1d_update weight → kt failed")?;
        let s_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(conv_state)
            .with_context(|| "kt-adapter: conv1d_update conv_state → kt failed")?;
        if !kiln_conv1d_kernel::supports_kt(&x_kt, &w_kt, &s_kt, kernel_size) {
            return Ok(None);
        }
        let out_kt = kiln_conv1d_kernel::causal_conv1d_update_kt(&x_kt, &w_kt, &s_kt, kernel_size)
            .map_err(|e| anyhow::anyhow!("kt causal_conv1d_update: {e}"))?;
        let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
            .with_context(|| "kt-adapter: conv1d_update out → candle failed")?;
        Ok(Some(out))
    }

    fn causal_conv1d_prefill(
        &self,
        x: &Tensor,
        weight: &Tensor,
        conv_state: &mut Tensor,
        kernel_size: usize,
    ) -> Result<Option<Tensor>> {
        if !self.fused_conv1d_enabled {
            return Ok(None);
        }
        // Phase 7 (#1082): kt-typed surface is now the only path.
        // See update path above for the borrow/in-place semantics.
        let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(x)
            .with_context(|| "kt-adapter: conv1d_prefill x → kt failed")?;
        let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(weight)
            .with_context(|| "kt-adapter: conv1d_prefill weight → kt failed")?;
        let s_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(conv_state)
            .with_context(|| "kt-adapter: conv1d_prefill conv_state → kt failed")?;
        if !kiln_conv1d_kernel::supports_prefill_kt(&x_kt, &w_kt, &s_kt, kernel_size) {
            return Ok(None);
        }
        let out_kt = kiln_conv1d_kernel::causal_conv1d_prefill_kt(&x_kt, &w_kt, &s_kt, kernel_size)
            .map_err(|e| anyhow::anyhow!("kt causal_conv1d_prefill: {e}"))?;
        let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
            .with_context(|| "kt-adapter: conv1d_prefill out → candle failed")?;
        Ok(Some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_backend() -> CudaBackend {
        CudaBackend {
            device: Device::Cpu,
            gdn_enabled: false,
            gdn_gates_enabled: false,
            gdn_gated_rms_norm_enabled: false,
            gdn_decode_fused_enabled: false,
            gdn_decode_unexpanded_qk_enabled: false,
            gdn_decode_qk_norm_recurrent_enabled: false,
            gdn_decode_qk_norm_recurrent_rmsnorm_enabled: false,
            fused_conv1d_enabled: false,
            cuda_use_kt_api_sgd_step: false,
            cuda_use_kt_api_adamw_step: false,
            lora_decode_add_enabled: false,
            gdn_full_chunk_forward_multiblock_enabled: false,
        }
    }

    #[test]
    fn cuda_resident_activation_registry_lifecycle() -> Result<()> {
        let backend = test_backend();
        let tensor = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;

        assert!(backend.supports_resident_activation());
        assert!(!backend.has_resident_activation(&tensor));

        backend.register_resident_activation(&tensor)?;
        assert!(backend.has_resident_activation(&tensor));

        backend.evict_resident_activation(&tensor);
        assert!(!backend.has_resident_activation(&tensor));

        backend.update_resident_activation(&tensor)?;
        assert!(backend.has_resident_activation(&tensor));

        Ok(())
    }

    #[test]
    fn cuda_optimizer_dispatch_declines_without_cuda_tensors() -> Result<()> {
        let backend = test_backend();
        let param = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
        let grad = Tensor::ones((2, 3), DType::F32, &Device::Cpu)?;
        let m = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
        let v = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;

        assert!(
            !backend.dispatch_sgd_step(&param, &grad, 0.01)?,
            "CUDA must not claim SGD dispatch for non-CUDA tensors"
        );
        assert!(
            !backend.dispatch_adamw_step(&param, &grad, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, 1)?,
            "CUDA must not claim AdamW dispatch for non-CUDA tensors"
        );

        backend.register_resident_activation(&param)?;
        backend.register_resident_activation(&grad)?;
        backend.register_resident_activation(&m)?;
        backend.register_resident_activation(&v)?;

        assert!(
            !backend.dispatch_sgd_step(&param, &grad, 0.01)?,
            "TensorId residency alone is not enough for CUDA to claim SGD ownership"
        );
        assert!(
            !backend.dispatch_adamw_step(&param, &grad, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, 1)?,
            "TensorId residency alone is not enough for CUDA to claim AdamW ownership"
        );

        Ok(())
    }

    #[test]
    fn cuda_sgd_step_resident_round_trip_f32() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!(
                    "CUDA unavailable, skipping cuda_sgd_step_resident_round_trip_f32: {err}"
                );
                return Ok(());
            }
        };
        let backend = CudaBackend::new(device.clone());
        let param = Tensor::from_slice(&[1.0f32, -2.0, 0.5, 3.0], (4,), &device)?;
        let grad = Tensor::from_slice(&[0.1f32, -0.2, 0.5, 1.0], (4,), &device)?;
        backend.register_resident_activation(&param)?;
        backend.register_resident_activation(&grad)?;

        assert!(backend.dispatch_sgd_step(&param, &grad, 0.25)?);
        let actual = param.to_vec1::<f32>()?;
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
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!(
                    "CUDA unavailable, skipping cuda_adamw_step_resident_round_trip_f32: {err}"
                );
                return Ok(());
            }
        };
        let backend = CudaBackend::new(device.clone());
        let param = Tensor::from_slice(&[1.0f32, -2.0, 0.5, 3.0], (4,), &device)?;
        let grad = Tensor::from_slice(&[0.5f32, -0.5, 0.25, -0.25], (4,), &device)?;
        let m = Tensor::zeros((4,), DType::F32, &device)?;
        let v = Tensor::zeros((4,), DType::F32, &device)?;
        backend.register_resident_activation(&param)?;
        backend.register_resident_activation(&grad)?;
        backend.register_resident_activation(&m)?;
        backend.register_resident_activation(&v)?;

        let lr = 0.01;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let weight_decay = 0.1;
        assert!(backend.dispatch_adamw_step(
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

        let actual = param.to_vec1::<f32>()?;
        let before = [1.0f32, -2.0, 0.5, 3.0];
        let grad_vals = [0.5f32, -0.5, 0.25, -0.25];
        for ((a, p0), g) in actual.iter().zip(before.iter()).zip(grad_vals.iter()) {
            let p_after_wd = *p0 * (1.0 - lr * weight_decay);
            let expected = p_after_wd - lr * (*g / (g.abs() + eps));
            assert!((a - expected).abs() < 1e-5, "actual={actual:?}");
        }
        let m_actual = m.to_vec1::<f32>()?;
        let v_actual = v.to_vec1::<f32>()?;
        for ((m_i, v_i), g) in m_actual.iter().zip(v_actual.iter()).zip(grad_vals.iter()) {
            assert!((m_i - (1.0 - beta1) * g).abs() < 1e-6);
            assert!((v_i - (1.0 - beta2) * g * g).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn cuda_sgd_and_adamw_resident_round_trip_bf16() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!(
                    "CUDA unavailable, skipping cuda_sgd_and_adamw_resident_round_trip_bf16: {err}"
                );
                return Ok(());
            }
        };
        let backend = CudaBackend::new(device.clone());

        let param =
            Tensor::from_slice(&[1.0f32, -2.0, 0.5, 3.0], (4,), &device)?.to_dtype(DType::BF16)?;
        let grad = Tensor::from_slice(&[0.25f32, -0.5, 0.5, -0.25], (4,), &device)?
            .to_dtype(DType::BF16)?;
        backend.register_resident_activation(&param)?;
        backend.register_resident_activation(&grad)?;
        assert!(backend.dispatch_sgd_step(&param, &grad, 0.5)?);
        let sgd_actual = param.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let sgd_expected = [0.875f32, -1.75, 0.25, 3.125];
        for (a, e) in sgd_actual.iter().zip(sgd_expected.iter()) {
            assert!(
                (a - e).abs() < 0.02,
                "actual={sgd_actual:?} expected={sgd_expected:?}"
            );
        }

        let adam_param =
            Tensor::from_slice(&[1.0f32, -2.0, 0.5, 3.0], (4,), &device)?.to_dtype(DType::BF16)?;
        let adam_grad = Tensor::from_slice(&[0.5f32, -0.5, 0.25, -0.25], (4,), &device)?
            .to_dtype(DType::BF16)?;
        let m = Tensor::zeros((4,), DType::BF16, &device)?;
        let v = Tensor::zeros((4,), DType::BF16, &device)?;
        backend.register_resident_activation(&adam_param)?;
        backend.register_resident_activation(&adam_grad)?;
        backend.register_resident_activation(&m)?;
        backend.register_resident_activation(&v)?;
        assert!(backend.dispatch_adamw_step(
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
        let adam_actual = adam_param.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let before = [1.0f32, -2.0, 0.5, 3.0];
        let grad_vals = [0.5f32, -0.5, 0.25, -0.25];
        for ((a, p0), g) in adam_actual.iter().zip(before.iter()).zip(grad_vals.iter()) {
            let expected = *p0 * (1.0 - 0.01 * 0.1) - 0.01 * (*g / (g.abs() + 1e-8));
            assert!((a - expected).abs() < 0.03, "actual={adam_actual:?}");
        }
        Ok(())
    }

    #[test]
    fn cuda_flash_attention_declines_tracked_training_tensors() -> Result<()> {
        let backend = test_backend();

        let q_base = Tensor::zeros((1, 2, 1, 128), DType::BF16, &Device::Cpu)?;
        let q_var = candle_core::Var::from_tensor(&q_base)?;
        let q = q_var.as_tensor();
        let k = Tensor::zeros((1, 2, 1, 128), DType::BF16, &Device::Cpu)?;
        let v = Tensor::zeros((1, 2, 1, 128), DType::BF16, &Device::Cpu)?;
        assert!(
            q.track_op(),
            "test precondition: q must be autograd-tracked"
        );
        assert!(
            backend.flash_attn_prefill(q, &k, &v, 1.0, true)?.is_none(),
            "CUDA FlashAttention prefill must decline tracked tensors until it has a bwd hook"
        );

        let q_decode_base = Tensor::zeros((1, 1, 1, 128), DType::BF16, &Device::Cpu)?;
        let q_decode_var = candle_core::Var::from_tensor(&q_decode_base)?;
        let q_decode = q_decode_var.as_tensor();
        let k_pool = Tensor::zeros((128, 1, 128), DType::BF16, &Device::Cpu)?;
        let v_pool = Tensor::zeros((128, 1, 128), DType::BF16, &Device::Cpu)?;
        let block_table = Tensor::zeros((1, 1), DType::U32, &Device::Cpu)?;
        let seqused_k = Tensor::zeros((1,), DType::I32, &Device::Cpu)?;

        assert!(
            backend
                .flash_attn_paged_decode(
                    q_decode,
                    &k_pool,
                    &v_pool,
                    &block_table,
                    128,
                    128,
                    1.0,
                    true
                )?
                .is_none(),
            "CUDA paged decode attention must decline tracked tensors"
        );
        assert!(
            backend
                .flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
                    q_decode,
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
            "CUDA dynamic paged decode attention must decline tracked tensors"
        );

        Ok(())
    }

    #[test]
    fn cuda_linear_prefill_apply_matches_candle_cuda_matmul() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!(
                    "CUDA unavailable, skipping cuda_linear_prefill_apply_matches_candle_cuda_matmul: {err}"
                );
                return Ok(());
            }
        };
        let backend = CudaBackend::new(device.clone());

        let x = Tensor::from_slice(&[1.0f32, -2.0, 0.5, 3.0, 4.0, -1.0], (2, 3), &device)?;
        let w = Tensor::from_slice(
            &[
                0.5f32, 1.0, -1.5, 2.0, -0.25, 0.75, 1.25, -0.5, 2.0, -1.0, 0.0, 0.5,
            ],
            (3, 4),
            &device,
        )?;

        let routed = backend
            .linear_prefill_apply(&x, &w)?
            .expect("CUDA linear_prefill_apply should accept CUDA tensors");
        let expected = x.broadcast_matmul(&w)?;
        assert_eq!(routed.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn cuda_linear_prefill_apply_offset_matches_candle_cuda_chunk() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!(
                    "CUDA unavailable, skipping cuda_linear_prefill_apply_offset_matches_candle_cuda_chunk: {err}"
                );
                return Ok(());
            }
        };
        let backend = CudaBackend::new(device.clone());

        let x = Tensor::from_slice(&[1.0f32, -2.0, 0.5, 3.0, 4.0, -1.0], (2, 3), &device)?;
        let w = Tensor::from_slice(
            &[
                0.5f32, 1.0, -1.5, 2.0, 3.0, -0.25, 0.75, 1.25, -0.5, 0.25, 2.0, -1.0, 0.0, 0.5,
                -2.0,
            ],
            (3, 5),
            &device,
        )?;

        let routed = backend
            .linear_prefill_apply_offset(&x, &w, 1, 3)?
            .expect("CUDA linear_prefill_apply_offset should accept CUDA tensors");
        let expected_chunk = w.narrow(1, 1, 3)?.contiguous()?;
        let expected = x.broadcast_matmul(&expected_chunk)?;
        assert_eq!(routed.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn cuda_registered_lora_delta_matches_candle_cuda_reference() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!(
                    "CUDA unavailable, skipping cuda_registered_lora_delta_matches_candle_cuda_reference: {err}"
                );
                return Ok(());
            }
        };
        let backend = CudaBackend::new(device.clone());

        let x = Tensor::from_slice(&[0.5f32, -1.0, 2.0, 1.5, 0.25, -0.75], (2, 3), &device)?;
        let a = Tensor::from_slice(&[0.25f32, -0.5, 1.0, 1.5, 0.0, -1.0], (2, 3), &device)?;
        let b = Tensor::from_slice(
            &[1.0f32, -0.25, 0.5, 0.75, -1.0, 0.25, 0.0, 1.5],
            (4, 2),
            &device,
        )?;
        let scale = 0.5;

        assert!(backend.lora_delta_resident(&x, &a, &b, scale)?.is_none());
        backend.register_resident_activation(&a)?;
        backend.register_resident_activation(&b)?;

        let routed = backend
            .lora_delta_resident(&x, &a, &b, scale)?
            .expect("registered CUDA LoRA delta should engage");
        let expected = compute_lora_delta(
            &x,
            &LoraProjectionWeights {
                a: a.clone(),
                b: b.clone(),
            },
            scale,
        )?;

        assert_eq!(routed.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        Ok(())
    }
}
