//! Candle-based forward pass layers for Qwen3.5-4B.
//!
//! Implements the foundational compute primitives: embedding lookup, RMSNorm,
//! RoPE (rotary position embeddings), and SwiGLU FFN. These operate on candle
//! `Tensor` objects and are composed into the full transformer forward pass.

use anyhow::{Context, Result};
use candle_core::backend::BackendDevice;
#[cfg(feature = "cuda")]
use candle_core::backend::BackendStorage;
#[cfg(feature = "cuda")]
use candle_core::op::BackpropOp;
#[cfg(feature = "cuda")]
use candle_core::{CpuStorage, CudaStorage, CustomOp2, CustomOp3, Layout, Shape, Storage};
use candle_core::{DType, Device, Tensor};
use std::cell::Cell;
use std::sync::{Mutex, OnceLock};

use crate::backend::BackendRuntime;
use crate::kv_cache::KvCache;
use crate::lora_loader::{
    LoraLayerWeights, LoraProjectionWeights, LoraWeights, compute_lora_delta, linear_with_lora_t,
};
use crate::paged_kv_cache::{PagedKvCache, contiguous_slot_run_start};
use crate::transposed_weight_cache::{
    CachedTransposedWeightBytes, transposed_weight_bytes_2d_cached_bytes,
};
use crate::weights::{DeferredMtpSource, ModelWeights, MtpWeights, TensorDType, WeightTensor};

use kiln_core::block::BlockTable;

// NVTX is always linked: when the `nvtx` cargo feature is off the
// `kiln_nvtx::range!` macro expands to a zero-sized RAII guard whose drop is
// a no-op (verified by the optimizer in release). This keeps the call sites
// below free of `#[cfg(feature = "nvtx")]` noise.

/// CUDA-compatible sigmoid: `1 / (1 + exp(-x))`.
///
/// `candle_nn::ops::sigmoid` lacks a CUDA kernel, so we implement it using
/// basic tensor operations that all have CUDA support.
fn cuda_sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg().context("cuda_sigmoid x.neg")?;
    let exp_neg_x = neg_x.exp().context("cuda_sigmoid exp")?;
    let one_plus = (exp_neg_x + 1.0).context("cuda_sigmoid add one")?;
    let result = one_plus.recip().context("cuda_sigmoid recip")?;
    Ok(result)
}

fn fused_paged_decode_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("KILN_DISABLE_FUSED_PAGED_DECODE").is_ok())
}

fn cuda_direct_paged_decode_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("KILN_DISABLE_CUDA_DIRECT_PAGED_DECODE").is_ok())
}

#[cfg(feature = "cuda")]
fn cuda_fused_rotary_qk_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("KILN_DISABLE_FUSED_CUDA_ROTARY_QK").is_ok())
}

#[cfg(feature = "cuda")]
fn cuda_fused_attn_decode_qkv_prep_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("KILN_DISABLE_CUDA_ATTN_DECODE_QKV_PREP").is_ok())
}

#[cfg(feature = "cuda")]
fn cuda_fused_mlp_silu_mul_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("KILN_DISABLE_FUSED_CUDA_MLP_SILU_MUL").is_ok())
}

#[cfg(feature = "cuda")]
fn cuda_fused_attn_sigmoid_mul_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("KILN_DISABLE_FUSED_CUDA_ATTN_SIGMOID_MUL").is_ok())
}

thread_local! {
    static VULKAN_SKIP_GDN_STATE_READBACK_DEPTH: Cell<usize> = const { Cell::new(0) };
}

#[allow(dead_code)]
pub(crate) fn vulkan_skip_gdn_state_readback_active() -> bool {
    VULKAN_SKIP_GDN_STATE_READBACK_DEPTH.with(|depth| depth.get() > 0)
}

pub(crate) struct VulkanSkipGdnStateReadbackScope {
    active: bool,
}

impl VulkanSkipGdnStateReadbackScope {
    pub(crate) fn new(active: bool) -> Self {
        if active {
            VULKAN_SKIP_GDN_STATE_READBACK_DEPTH.with(|depth| depth.set(depth.get() + 1));
        }
        Self { active }
    }
}

impl Drop for VulkanSkipGdnStateReadbackScope {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        VULKAN_SKIP_GDN_STATE_READBACK_DEPTH.with(|depth| {
            let previous = depth.get();
            debug_assert!(previous > 0);
            depth.set(previous.saturating_sub(1));
        });
    }
}

/// Threshold above which the fused `kiln_rmsnorm_kernel::fused_rmsnorm_with_autograd`
/// CustomOp2 path is enabled by default during training. Set to 47 GiB to draw the
/// line between A6000-class GPUs (49 140 MiB) and A40-class GPUs (46 068 MiB).
///
/// See `docs/audits/PHASE10_VRAM_REGRESSION_MECHANISM.md` (PR #643) — the
/// CustomOp2 saved-tensor expansion costs +18.6 GiB peak at T=2048 on A40 but is
/// invisibly absorbed by the larger allocator-pool baseline that A6000 sits on
/// permanently. Gating the default path on detected total VRAM keeps the fusion
/// savings on production hardware while protecting smaller GPUs from OOM.
#[cfg(feature = "cuda")]
const FUSED_RMSNORM_VRAM_GATE_BYTES: u64 = 47 * 1024 * 1024 * 1024;

/// Decide whether the fused RMSNorm CustomOp2 path should be the default
/// dispatch on this CUDA host. Computed exactly once per process via OnceLock
/// so the (potentially shell-out) VRAM detection is amortized away from the
/// training inner loop.
///
/// Returns `true` (default-on, fused path) when one of the following holds:
///   * `KILN_FORCE_RMSNORM_KERNEL=1` is set (debug/benchmark override that
///     bypasses the gate even on small GPUs — useful for reproducing the A40
///     +18.6 GiB regression locally).
///   * Detected total VRAM is at least `FUSED_RMSNORM_VRAM_GATE_BYTES` (47 GiB).
///
/// Returns `false` (gated off, fall through to `rms_norm_fallback`) when:
///   * VRAM detection failed (safer to assume small GPU).
///   * Detected total VRAM is below the gate threshold.
///
/// Emits a single `tracing::info!` line at first call documenting the inputs to
/// the decision, so a single `grep "kiln rmsnorm gate"` on a training log
/// answers "which path is this run on?".
///
/// The hard kill switches `KILN_DISABLE_RMSNORM_KERNEL` and
/// `KILN_DISABLE_RMSNORM_BACKWARD` are checked separately at the dispatch site
/// in `rms_norm()` and take precedence over the gate.
#[cfg(feature = "cuda")]
fn should_use_fused_rmsnorm() -> bool {
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(|| {
        let force = std::env::var("KILN_FORCE_RMSNORM_KERNEL").is_ok();
        let vram = kiln_core::vram::detect_vram();
        let total_bytes = vram.total_bytes;
        let total_mib = total_bytes / (1024 * 1024);
        let threshold_mib = FUSED_RMSNORM_VRAM_GATE_BYTES / (1024 * 1024);
        let detected_meets_threshold = total_bytes >= FUSED_RMSNORM_VRAM_GATE_BYTES;
        let take_fused = force || detected_meets_threshold;
        tracing::info!(
            total_vram_mib = total_mib,
            threshold_mib = threshold_mib,
            detection_source = %vram.source,
            force_override = force,
            fused_path = if take_fused { "ON" } else { "OFF" },
            "kiln rmsnorm gate"
        );
        take_fused
    })
}

/// CUDA-compatible SiLU (Swish): `x * sigmoid(x)`.
fn cuda_silu(x: &Tensor) -> Result<Tensor> {
    let sig = cuda_sigmoid(x)?;
    Ok((x * sig)?)
}

fn any_tensor_tracks_op(tensors: &[&Tensor]) -> bool {
    tensors.iter().any(|tensor| tensor.track_op())
}

fn env_truthy_for_profile(name: &str) -> bool {
    env_truthy(name)
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

#[cfg(feature = "metal")]
fn metal_streaming_gdn_forward_only_fastpaths_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy("KILN_ENABLE_METAL_GDN_STREAMING_FASTPATHS"))
}

fn streaming_gdn_forward_only_fastpaths_allowed(_device: &Device) -> bool {
    #[cfg(feature = "metal")]
    {
        if matches!(_device, Device::Metal(_)) {
            return metal_streaming_gdn_forward_only_fastpaths_enabled();
        }
    }
    true
}

fn profile_paged_layers_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy_for_profile("KILN_PROFILE_PAGED_LAYERS"))
}

fn profile_gdn_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy_for_profile("KILN_PROFILE_GDN_STAGES"))
}

fn profile_gdn_recurrent_inner_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy_for_profile("KILN_PROFILE_GDN_RECURRENT_INNER_STAGES"))
}

fn profile_full_attn_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy_for_profile("KILN_PROFILE_FULL_ATTN_STAGES"))
}

fn profile_mlp_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy_for_profile("KILN_PROFILE_MLP_STAGES"))
}

#[cfg(feature = "cuda")]
fn cuda_gdn_ab_in_proj_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_CUDA_GDN_AB_IN_PROJ").is_err())
}

#[cfg(feature = "cuda")]
fn cuda_gdn_prefill_ab_in_proj_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_CUDA_GDN_PREFILL_AB_IN_PROJ").is_err())
}

#[cfg(feature = "cuda")]
const CUDA_GDN_PREFILL_AB_IN_PROJ_MAX_TOKENS: usize = 128;

#[cfg(feature = "cuda")]
fn cuda_full_attn_qkv_in_proj_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_CUDA_FULL_ATTN_QKV_IN_PROJ").is_err())
}

fn weighted_lm_head_prep_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| env_truthy_for_profile("KILN_DISABLE_WEIGHTED_LM_HEAD_PREP"))
}

fn vulkan_gdn_recurrent_step_f32_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_STEP_F32").is_err())
}

fn synchronize_for_profile(device: &Device) -> Result<()> {
    if let Device::Metal(device) = device {
        device.synchronize()?;
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn metal_autoreleasepool<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    objc2::rc::autoreleasepool(|_| f())
}

fn log_paged_layer_profile(
    layer: usize,
    kind: &str,
    seq_len: usize,
    start_pos: usize,
    elapsed: std::time::Duration,
) {
    eprintln!(
        "kiln_profile_paged_layer layer={layer} kind={kind} seq_len={seq_len} start_pos={start_pos} elapsed_ms={:.3}",
        elapsed.as_secs_f64() * 1000.0
    );
}

fn start_gdn_stage_profile(
    device: &Device,
    context: Option<(usize, usize)>,
) -> Result<Option<std::time::Instant>> {
    if context.is_some() {
        synchronize_for_profile(device)?;
        Ok(Some(std::time::Instant::now()))
    } else {
        Ok(None)
    }
}

fn finish_gdn_stage_profile(
    device: &Device,
    context: Option<(usize, usize)>,
    stage: &str,
    seq_len: usize,
    start: Option<std::time::Instant>,
) -> Result<()> {
    let Some(start) = start else {
        return Ok(());
    };
    let Some((layer, start_pos)) = context else {
        return Ok(());
    };
    synchronize_for_profile(device)?;
    eprintln!(
        "kiln_profile_gdn_stage layer={layer} stage={stage} seq_len={seq_len} start_pos={start_pos} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

fn start_gdn_recurrent_inner_profile(
    device: &Device,
    enabled: bool,
) -> Result<Option<std::time::Instant>> {
    if enabled {
        synchronize_for_profile(device)?;
        Ok(Some(std::time::Instant::now()))
    } else {
        Ok(None)
    }
}

fn finish_gdn_recurrent_inner_profile(
    device: &Device,
    stage: &str,
    batch: usize,
    heads: usize,
    seq_len: usize,
    chunk_index: usize,
    chunk_len: usize,
    start: Option<std::time::Instant>,
) -> Result<()> {
    let Some(start) = start else {
        return Ok(());
    };
    synchronize_for_profile(device)?;
    eprintln!(
        "kiln_profile_gdn_recurrent_inner_stage stage={stage} batch={batch} heads={heads} seq_len={seq_len} chunk_index={chunk_index} chunk_len={chunk_len} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

fn start_full_attn_stage_profile(
    device: &Device,
    context: Option<(usize, usize)>,
) -> Result<Option<std::time::Instant>> {
    if context.is_some() {
        synchronize_for_profile(device)?;
        Ok(Some(std::time::Instant::now()))
    } else {
        Ok(None)
    }
}

fn start_named_full_attn_stage_profile(
    device: &Device,
    context: Option<(usize, usize)>,
    stage: &str,
    seq_len: usize,
) -> Result<Option<std::time::Instant>> {
    let start = start_full_attn_stage_profile(device, context)?;
    if let Some((full_attn_layer, start_pos)) = context {
        eprintln!(
            "kiln_profile_full_attn_stage_begin full_attn_layer={full_attn_layer} stage={stage} seq_len={seq_len} start_pos={start_pos}"
        );
    }
    Ok(start)
}

fn finish_full_attn_stage_profile(
    device: &Device,
    context: Option<(usize, usize)>,
    stage: &str,
    seq_len: usize,
    start: Option<std::time::Instant>,
) -> Result<()> {
    let Some(start) = start else {
        return Ok(());
    };
    let Some((full_attn_layer, start_pos)) = context else {
        return Ok(());
    };
    synchronize_for_profile(device)?;
    eprintln!(
        "kiln_profile_full_attn_stage full_attn_layer={full_attn_layer} stage={stage} seq_len={seq_len} start_pos={start_pos} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

fn start_mlp_stage_profile(
    device: &Device,
    context: Option<(usize, usize)>,
) -> Result<Option<std::time::Instant>> {
    if context.is_some() {
        synchronize_for_profile(device)?;
        Ok(Some(std::time::Instant::now()))
    } else {
        Ok(None)
    }
}

fn finish_mlp_stage_profile(
    device: &Device,
    context: Option<(usize, usize)>,
    stage: &str,
    seq_len: usize,
    start: Option<std::time::Instant>,
) -> Result<()> {
    let Some(start) = start else {
        return Ok(());
    };
    let Some((layer, start_pos)) = context else {
        return Ok(());
    };
    synchronize_for_profile(device)?;
    eprintln!(
        "kiln_profile_mlp_stage layer={layer} stage={stage} seq_len={seq_len} start_pos={start_pos} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

#[cfg(feature = "metal")]
fn try_metal_mlp_gate_up_hidden(
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora_layer: Option<&LoraLayerWeights>,
) -> Result<Option<Tensor>> {
    if lora_layer.is_some_and(LoraLayerWeights::has_mlp_gate_up)
        || crate::mtp_debug::is_mtp_fp32_head_armed()
        || crate::backend::metal::metal_mlp_gate_up_fusion_disabled()
    {
        return Ok(None);
    }
    if mlp.gate_proj_marlin.is_some()
        || mlp.up_proj_marlin.is_some()
        || mlp.down_proj_marlin.is_some()
    {
        return Ok(None);
    }
    if !crate::backend::metal::metal_mlp_gate_up_supports(x, &mlp.gate_proj_t, &mlp.up_proj_t) {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/mlp/gate_up_fused");
    Ok(Some(crate::backend::metal::metal_mlp_gate_up_bf16(
        x,
        &mlp.gate_proj_t,
        &mlp.up_proj_t,
    )?))
}

fn linear_with_lora_t_decode(
    x: &Tensor,
    weight_t: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    {
        if !crate::mtp_debug::is_mtp_fp32_head_armed()
            && !crate::mtp_debug::is_mtp_single_token_self_attn_armed()
            && (crate::backend::metal::metal_transposed_coop_gemv_supports(x, weight_t)
                || crate::backend::metal::metal_transposed_coop_gemv_decode_batch_supports(
                    x, weight_t,
                ))
        {
            let base = crate::backend::metal::metal_transposed_coop_gemv_bf16(x, weight_t)
                .context("metal transposed coop GEMV failed");
            return add_lora_delta_to_base(None, base?, x, lora, lora_scale)
                .context("metal transposed coop GEMV LoRA delta failed");
        }
    }

    linear_with_lora_t(x, weight_t, lora, lora_scale)
}

fn add_lora_delta_to_base(
    backend: Option<&dyn BackendRuntime>,
    base: Tensor,
    x: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    let Some(proj) = lora else {
        return Ok(base);
    };
    #[cfg(feature = "cuda")]
    if let Some(out) = cuda_lora_add_training_f32(&base, x, proj, lora_scale)? {
        return Ok(out);
    }
    #[cfg(feature = "cuda")]
    if let Some(out) = cuda_lora_add_training_bf16(&base, x, proj, lora_scale)? {
        return Ok(out);
    }
    if let Some(backend) = backend {
        if let Some(out) = backend.lora_decode_add(&base, x, &proj.a, &proj.b, lora_scale)? {
            return Ok(out);
        }
        // Phase 4.1 step 2 + 5 (autograd-safe via CustomOp3): when
        // both A and B are registry-resident, dispatch the LoRA
        // delta on-device. The Vulkan impl wraps the dispatch in
        // `VulkanLoraOp` which provides analytic gradients for x,
        // A, and B — so this path is now safe to use during training
        // too. `loss.backward()` produces correct grad_A and grad_B
        // that flow into the SGD update.
        if let Some(delta) = backend.lora_delta_resident(x, &proj.a, &proj.b, lora_scale)? {
            let delta = if delta.dtype() == base.dtype() {
                delta
            } else {
                delta.to_dtype(base.dtype())?
            };
            return Ok((base + delta)?);
        }
    }
    #[cfg(feature = "metal")]
    {
        if crate::backend::metal::metal_lora_add_decode_supports(&base, x, &proj.a, &proj.b) {
            return crate::backend::metal::metal_lora_add_decode_bf16(
                &base, x, &proj.a, &proj.b, lora_scale,
            )
            .context("metal LoRA decode delta/add failed");
        }
    }
    let delta = compute_lora_delta(x, proj, lora_scale)?;
    let delta = if delta.dtype() == base.dtype() {
        delta
    } else {
        delta.to_dtype(base.dtype())?
    };
    Ok((base + delta)?)
}

#[cfg(feature = "cuda")]
fn cuda_lora_training_add_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("KILN_DISABLE_CUDA_LORA_TRAINING_ADD").is_ok())
}

#[cfg(feature = "cuda")]
fn cuda_lora_training_linear_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("KILN_DISABLE_CUDA_LORA_TRAINING_LINEAR").is_ok())
}

#[cfg(feature = "cuda")]
fn to_dtype_if_needed(t: &Tensor, dtype: DType) -> candle_core::Result<Tensor> {
    if t.dtype() == dtype {
        Ok(t.clone())
    } else {
        t.to_dtype(dtype)
    }
}

#[cfg(feature = "cuda")]
fn cuda_lora_bwd_tile_rows() -> usize {
    static TILE_ROWS: OnceLock<usize> = OnceLock::new();
    *TILE_ROWS.get_or_init(|| {
        std::env::var("KILN_CUDA_LORA_BWD_TILE_ROWS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(512)
    })
}

#[cfg(feature = "cuda")]
fn cuda_lora_linear_training_bf16(
    x: &Tensor,
    weight_t: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    scale: f32,
) -> Result<Option<Tensor>> {
    let Some(proj) = lora else {
        return Ok(None);
    };
    if cuda_lora_training_linear_disabled()
        || x.dtype() != DType::BF16
        || weight_t.dtype() != DType::BF16
        || !x.track_op()
        || !matches!(x.device(), Device::Cuda(_))
        || !matches!(weight_t.device(), Device::Cuda(_))
    {
        return Ok(None);
    }

    let x_dims = x.dims();
    if x_dims.len() < 2 {
        return Ok(None);
    }
    let Ok((in_features, out_dim)) = weight_t.dims2() else {
        return Ok(None);
    };
    if *x_dims.last().unwrap() != in_features {
        return Ok(None);
    }
    let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
    if rows == 0 {
        let mut out_dims = x_dims.to_vec();
        *out_dims.last_mut().unwrap() = out_dim;
        return Ok(Some(Tensor::zeros(out_dims, DType::BF16, x.device())?));
    }

    let Ok((rank, a_in)) = proj.a.dims2() else {
        return Ok(None);
    };
    let Ok((b_out, b_rank)) = proj.b.dims2() else {
        return Ok(None);
    };
    if a_in != in_features || b_out != out_dim || b_rank != rank {
        return Ok(None);
    }

    let x_2d = x.reshape((rows, in_features))?;
    if !x_2d.is_contiguous() {
        return Ok(None);
    }
    let a_bf16 = to_dtype_if_needed(&proj.a, DType::BF16)?.contiguous()?;
    let b_bf16 = to_dtype_if_needed(&proj.b, DType::BF16)?.contiguous()?;
    let out_2d = x_2d
        .apply_op3(
            &a_bf16,
            &b_bf16,
            CudaLoraLinearBf16 {
                weight_t: weight_t.clone(),
                scale,
            },
        )
        .context("cuda BF16 LoRA training fused linear")?;
    let mut out_dims = x_dims.to_vec();
    *out_dims.last_mut().unwrap() = out_dim;
    Ok(Some(out_2d.reshape(out_dims)?))
}

#[cfg(feature = "cuda")]
fn cuda_lora_add_training_f32(
    base: &Tensor,
    x: &Tensor,
    proj: &LoraProjectionWeights,
    scale: f32,
) -> Result<Option<Tensor>> {
    if cuda_lora_training_add_disabled()
        || base.dtype() != DType::F32
        || x.dtype() != DType::F32
        || !matches!(base.device(), Device::Cuda(_))
        || !matches!(x.device(), Device::Cuda(_))
    {
        return Ok(None);
    }

    let base_dims = base.dims();
    let x_dims = x.dims();
    if base_dims.len() < 2 || x_dims.len() != base_dims.len() {
        return Ok(None);
    }
    if base_dims[..base_dims.len() - 1] != x_dims[..x_dims.len() - 1] {
        return Ok(None);
    }
    let out_dim = *base_dims.last().unwrap();
    let in_features = *x_dims.last().unwrap();
    let rows: usize = base_dims[..base_dims.len() - 1].iter().product();
    if rows == 0 {
        return Ok(Some(base.clone()));
    }

    let Ok((rank, a_in)) = proj.a.dims2() else {
        return Ok(None);
    };
    let Ok((b_out, b_rank)) = proj.b.dims2() else {
        return Ok(None);
    };
    if a_in != in_features || b_out != out_dim || b_rank != rank {
        return Ok(None);
    }

    let base_2d = base.reshape((rows, out_dim))?;
    let x_2d = x.reshape((rows, in_features))?;
    if !base_2d.is_contiguous() || !x_2d.is_contiguous() {
        return Ok(None);
    }

    let a_f32 = proj.a.to_dtype(DType::F32)?.contiguous()?;
    let b_f32 = proj.b.to_dtype(DType::F32)?.contiguous()?;
    let a_t = a_f32.t()?.contiguous()?;
    let hidden = x_2d
        .matmul(&a_t)
        .context("cuda LoRA training add hidden matmul")?
        .contiguous()
        .context("cuda LoRA training add hidden contiguous")?;
    let out_2d = base_2d
        .apply_op3(&hidden, &b_f32, CudaLoraAddF32 { scale })
        .context("cuda LoRA training add CustomOp3")?;
    Ok(Some(out_2d.reshape(base_dims)?))
}

#[cfg(feature = "cuda")]
fn cuda_lora_add_training_bf16(
    base: &Tensor,
    x: &Tensor,
    proj: &LoraProjectionWeights,
    scale: f32,
) -> Result<Option<Tensor>> {
    if cuda_lora_training_add_disabled()
        || base.dtype() != DType::BF16
        || x.dtype() != DType::BF16
        || !matches!(base.device(), Device::Cuda(_))
        || !matches!(x.device(), Device::Cuda(_))
    {
        return Ok(None);
    }

    let base_dims = base.dims();
    let x_dims = x.dims();
    if base_dims.len() < 2 || x_dims.len() != base_dims.len() {
        return Ok(None);
    }
    if base_dims[..base_dims.len() - 1] != x_dims[..x_dims.len() - 1] {
        return Ok(None);
    }
    let out_dim = *base_dims.last().unwrap();
    let in_features = *x_dims.last().unwrap();
    let rows: usize = base_dims[..base_dims.len() - 1].iter().product();
    if rows == 0 {
        return Ok(Some(base.clone()));
    }

    let Ok((rank, a_in)) = proj.a.dims2() else {
        return Ok(None);
    };
    let Ok((b_out, b_rank)) = proj.b.dims2() else {
        return Ok(None);
    };
    if a_in != in_features || b_out != out_dim || b_rank != rank {
        return Ok(None);
    }

    let base_2d = base.reshape((rows, out_dim))?;
    let x_2d = x.reshape((rows, in_features))?;
    if !base_2d.is_contiguous() || !x_2d.is_contiguous() {
        return Ok(None);
    }

    let a_bf16 = proj.a.to_dtype(DType::BF16)?.contiguous()?;
    let b_bf16 = proj.b.to_dtype(DType::BF16)?.contiguous()?;
    let a_t = a_bf16.t()?.contiguous()?;
    let hidden = x_2d
        .matmul(&a_t)
        .context("cuda BF16 LoRA training add hidden matmul")?
        .to_dtype(DType::F32)?
        .contiguous()
        .context("cuda BF16 LoRA training add hidden contiguous")?;
    let out_2d = base_2d
        .apply_op3(&hidden, &b_bf16, CudaLoraAddBf16 { scale })
        .context("cuda BF16 LoRA training add CustomOp3")?;
    Ok(Some(out_2d.reshape(base_dims)?))
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
struct CudaLoraAddF32 {
    scale: f32,
}

#[cfg(feature = "cuda")]
impl CustomOp3 for CudaLoraAddF32 {
    fn name(&self) -> &'static str {
        "kiln-cuda-lora-add-f32"
    }

    fn cpu_fwd(
        &self,
        s_base: &CpuStorage,
        l_base: &Layout,
        s_hidden: &CpuStorage,
        l_hidden: &Layout,
        s_b: &CpuStorage,
        l_b: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        if !l_base.is_contiguous()
            || !l_hidden.is_contiguous()
            || !l_b.is_contiguous()
            || l_base.start_offset() != 0
            || l_hidden.start_offset() != 0
            || l_b.start_offset() != 0
        {
            candle_core::bail!("CudaLoraAddF32 CPU fallback requires compact contiguous inputs");
        }
        let base = Tensor::from_storage(
            Storage::Cpu(s_base.clone()),
            Shape::from(l_base.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let hidden = Tensor::from_storage(
            Storage::Cpu(s_hidden.clone()),
            Shape::from(l_hidden.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let b = Tensor::from_storage(
            Storage::Cpu(s_b.clone()),
            Shape::from(l_b.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let delta = (hidden.matmul(&b.t()?)? * self.scale as f64)?;
        let out = (base + delta)?;
        let (storage, layout) = out.storage_and_layout();
        let storage = storage.try_clone(layout)?;
        match storage {
            Storage::Cpu(storage) => Ok((storage, Shape::from(l_base.dims().to_vec()))),
            _ => candle_core::bail!("CudaLoraAddF32 CPU fallback produced non-CPU storage"),
        }
    }

    fn cuda_fwd(
        &self,
        s_base: &CudaStorage,
        l_base: &Layout,
        s_hidden: &CudaStorage,
        l_hidden: &Layout,
        s_b: &CudaStorage,
        l_b: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        if !l_base.is_contiguous() || !l_hidden.is_contiguous() || !l_b.is_contiguous() {
            candle_core::bail!("CudaLoraAddF32 CUDA path requires contiguous inputs");
        }
        let out_storage = s_base.try_clone(l_base)?;
        let out_shape = Shape::from(l_base.dims().to_vec());
        let out_layout = Layout::contiguous(out_shape.clone());
        kiln_rmsnorm_kernel::lora_add_inplace_f32_storage(
            &out_storage,
            &out_layout,
            s_hidden,
            l_hidden,
            s_b,
            l_b,
            self.scale,
        )
        .map_err(|e| candle_core::Error::Msg(format!("CudaLoraAddF32 CUDA add: {e:?}")))?;
        Ok((out_storage, out_shape))
    }

    fn bwd(
        &self,
        _base: &Tensor,
        hidden: &Tensor,
        b: &Tensor,
        _res: &Tensor,
        grad_y: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let grad_base = grad_y.clone();
        let grad_y_f32 = to_dtype_if_needed(grad_y, DType::F32)?;
        let grad_delta = (grad_y_f32 * self.scale as f64)?;
        let b_f32 = to_dtype_if_needed(b, DType::F32)?;
        let hidden_f32 = to_dtype_if_needed(hidden, DType::F32)?;
        let grad_hidden = grad_delta.matmul(&b_f32)?;
        let grad_b = grad_delta.t()?.contiguous()?.matmul(&hidden_f32)?;
        Ok((Some(grad_base), Some(grad_hidden), Some(grad_b)))
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
struct CudaLoraAddBf16 {
    scale: f32,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct CudaLoraLinearBf16 {
    weight_t: Tensor,
    scale: f32,
}

#[cfg(feature = "cuda")]
impl CudaLoraLinearBf16 {
    fn forward_tensor(&self, x: &Tensor, a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
        let base = x.matmul(&self.weight_t)?;
        let a_t = a.t()?.contiguous()?;
        let hidden = x.matmul(&a_t)?.to_dtype(DType::F32)?.contiguous()?;
        base.apply_op3_no_bwd(&hidden, b, &CudaLoraAddBf16 { scale: self.scale })
    }
}

#[cfg(feature = "cuda")]
impl CustomOp3 for CudaLoraLinearBf16 {
    fn name(&self) -> &'static str {
        "kiln-cuda-lora-linear-bf16"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
        s_a: &CpuStorage,
        l_a: &Layout,
        s_b: &CpuStorage,
        l_b: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        if !l_x.is_contiguous()
            || !l_a.is_contiguous()
            || !l_b.is_contiguous()
            || l_x.start_offset() != 0
            || l_a.start_offset() != 0
            || l_b.start_offset() != 0
        {
            candle_core::bail!(
                "CudaLoraLinearBf16 CPU fallback requires compact contiguous inputs"
            );
        }
        let x = Tensor::from_storage(
            Storage::Cpu(s_x.clone()),
            Shape::from(l_x.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let a = Tensor::from_storage(
            Storage::Cpu(s_a.clone()),
            Shape::from(l_a.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let b = Tensor::from_storage(
            Storage::Cpu(s_b.clone()),
            Shape::from(l_b.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let out = self.forward_tensor(&x, &a, &b)?;
        let (storage, layout) = out.storage_and_layout();
        let storage = storage.try_clone(layout)?;
        match storage {
            Storage::Cpu(storage) => Ok((storage, Shape::from(out.dims().to_vec()))),
            _ => candle_core::bail!("CudaLoraLinearBf16 CPU fallback produced non-CPU storage"),
        }
    }

    fn cuda_fwd(
        &self,
        s_x: &CudaStorage,
        l_x: &Layout,
        s_a: &CudaStorage,
        l_a: &Layout,
        s_b: &CudaStorage,
        l_b: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        if !l_x.is_contiguous() || !l_a.is_contiguous() || !l_b.is_contiguous() {
            candle_core::bail!("CudaLoraLinearBf16 CUDA path requires contiguous inputs");
        }
        let x_dims = l_x.dims();
        let a_dims = l_a.dims();
        let b_dims = l_b.dims();
        if x_dims.len() != 2 || a_dims.len() != 2 || b_dims.len() != 2 {
            candle_core::bail!(
                "CudaLoraLinearBf16 CUDA path expects rank-2 inputs, got x={x_dims:?} a={a_dims:?} b={b_dims:?}"
            );
        }
        let (rows, in_features) = (x_dims[0], x_dims[1]);
        let (rank, a_in) = (a_dims[0], a_dims[1]);
        let (out_dim, b_rank) = (b_dims[0], b_dims[1]);
        if a_in != in_features || b_rank != rank {
            candle_core::bail!(
                "CudaLoraLinearBf16 CUDA shape mismatch x={x_dims:?} a={a_dims:?} b={b_dims:?}"
            );
        }
        let (weight_storage, weight_layout) = self.weight_t.storage_and_layout();
        let Storage::Cuda(weight_storage) = &*weight_storage else {
            candle_core::bail!("CudaLoraLinearBf16 CUDA path requires CUDA weight storage");
        };
        if weight_layout.dims() != [in_features, out_dim] {
            candle_core::bail!(
                "CudaLoraLinearBf16 CUDA weight shape mismatch weight={:?} x={x_dims:?} b={b_dims:?}",
                weight_layout.dims()
            );
        }

        let out_shape = Shape::from(vec![rows, out_dim]);
        let out_layout = Layout::contiguous(out_shape.clone());
        let out_storage = s_x.matmul(
            weight_storage,
            (1, rows, out_dim, in_features),
            l_x,
            weight_layout,
        )?;

        let a_t_layout = l_a.transpose(0, 1)?;
        let hidden_bf16 = s_x.matmul(s_a, (1, rows, rank, in_features), l_x, &a_t_layout)?;
        let hidden_shape = Shape::from(vec![rows, rank]);
        let hidden_layout = Layout::contiguous(hidden_shape);
        let hidden_f32 = hidden_bf16.to_dtype(&hidden_layout, DType::F32)?;
        kiln_rmsnorm_kernel::lora_add_bf16_storage(
            &out_storage,
            &out_layout,
            &out_storage,
            &out_layout,
            &hidden_f32,
            &hidden_layout,
            s_b,
            l_b,
            self.scale,
        )
        .map_err(|e| candle_core::Error::Msg(format!("CudaLoraLinearBf16 CUDA add: {e:?}")))?;
        Ok((out_storage, out_shape))
    }

    fn bwd(
        &self,
        x: &Tensor,
        a: &Tensor,
        b: &Tensor,
        _res: &Tensor,
        grad_y: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let rows = grad_y.dim(0)?;
        let tile_rows = cuda_lora_bwd_tile_rows().min(rows.max(1));
        let weight_t_t = self.weight_t.t()?;
        let a_bf16 = to_dtype_if_needed(a, DType::BF16)?;
        let a_t_bf16 = a_bf16.t()?.contiguous()?;
        let a_f32 = to_dtype_if_needed(a, DType::F32)?;
        let b_f32 = to_dtype_if_needed(b, DType::F32)?;
        let x_in = x.dim(1)?;
        let grad_x_shape = Shape::from(vec![rows, x_in]);
        let Device::Cuda(cuda_device) = x.device() else {
            candle_core::bail!("CudaLoraLinearBf16 backward requires CUDA input");
        };
        let mut grad_x_storage = unsafe { cuda_device.alloc_uninit(&grad_x_shape, DType::BF16)? };
        let mut grad_a_acc: Option<Tensor> = None;
        let mut grad_b_acc: Option<Tensor> = None;

        for start in (0..rows).step_by(tile_rows) {
            let len = (rows - start).min(tile_rows);
            let x_tile = x.narrow(0, start, len)?;
            let x_tile_f32 = to_dtype_if_needed(&x_tile, DType::F32)?;
            let grad_y_tile = grad_y.narrow(0, start, len)?;
            let grad_y_tile_bf16 = to_dtype_if_needed(&grad_y_tile, DType::BF16)?;
            let grad_y_tile_f32 = to_dtype_if_needed(&grad_y_tile, DType::F32)?;

            let grad_x_base = grad_y_tile_bf16.matmul(&weight_t_t)?;
            let grad_hidden = grad_y_tile_f32
                .matmul(&b_f32)?
                .affine(self.scale as f64, 0.0)?;
            let grad_x_lora = grad_hidden.matmul(&a_f32)?.to_dtype(DType::BF16)?;
            let grad_x_tile = (grad_x_base + grad_x_lora)?;
            let (grad_x_tile_storage, grad_x_tile_layout) = grad_x_tile.storage_and_layout();
            let Storage::Cuda(grad_x_tile_storage) = &*grad_x_tile_storage else {
                candle_core::bail!("CudaLoraLinearBf16 backward produced non-CUDA grad_x tile");
            };
            grad_x_tile_storage.copy2d(
                &mut grad_x_storage,
                len,
                x_in,
                x_in,
                x_in,
                grad_x_tile_layout.start_offset(),
                start * x_in,
            )?;

            let hidden = x_tile
                .matmul(&a_t_bf16)?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let grad_b_tile = grad_y_tile_f32
                .t()?
                .matmul(&hidden)?
                .affine(self.scale as f64, 0.0)?;
            grad_b_acc = Some(match grad_b_acc {
                Some(acc) => (acc + grad_b_tile)?,
                None => grad_b_tile,
            });

            let grad_a_tile = grad_hidden.t()?.matmul(&x_tile_f32)?;
            grad_a_acc = Some(match grad_a_acc {
                Some(acc) => (acc + grad_a_tile)?,
                None => grad_a_tile,
            });
        }

        let grad_x = Tensor::from_storage(
            Storage::Cuda(grad_x_storage),
            grad_x_shape,
            BackpropOp::none(),
            false,
        );
        let Some(grad_a_acc) = grad_a_acc else {
            candle_core::bail!("CudaLoraLinearBf16 backward produced no A gradient tiles");
        };
        let Some(grad_b_acc) = grad_b_acc else {
            candle_core::bail!("CudaLoraLinearBf16 backward produced no B gradient tiles");
        };
        let grad_a = grad_a_acc.to_dtype(a.dtype())?;
        let grad_b = grad_b_acc.to_dtype(b.dtype())?;
        Ok((Some(grad_x), Some(grad_a), Some(grad_b)))
    }
}

#[cfg(feature = "cuda")]
impl CustomOp3 for CudaLoraAddBf16 {
    fn name(&self) -> &'static str {
        "kiln-cuda-lora-add-bf16"
    }

    fn cpu_fwd(
        &self,
        s_base: &CpuStorage,
        l_base: &Layout,
        s_hidden: &CpuStorage,
        l_hidden: &Layout,
        s_b: &CpuStorage,
        l_b: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        if !l_base.is_contiguous()
            || !l_hidden.is_contiguous()
            || !l_b.is_contiguous()
            || l_base.start_offset() != 0
            || l_hidden.start_offset() != 0
            || l_b.start_offset() != 0
        {
            candle_core::bail!("CudaLoraAddBf16 CPU fallback requires compact contiguous inputs");
        }
        let base = Tensor::from_storage(
            Storage::Cpu(s_base.clone()),
            Shape::from(l_base.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let hidden = Tensor::from_storage(
            Storage::Cpu(s_hidden.clone()),
            Shape::from(l_hidden.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let b = Tensor::from_storage(
            Storage::Cpu(s_b.clone()),
            Shape::from(l_b.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let delta = (hidden
            .to_dtype(DType::F32)?
            .matmul(&b.to_dtype(DType::F32)?.t()?)?
            * self.scale as f64)?;
        let out = (base.to_dtype(DType::F32)? + delta)?.to_dtype(DType::BF16)?;
        let (storage, layout) = out.storage_and_layout();
        let storage = storage.try_clone(layout)?;
        match storage {
            Storage::Cpu(storage) => Ok((storage, Shape::from(l_base.dims().to_vec()))),
            _ => candle_core::bail!("CudaLoraAddBf16 CPU fallback produced non-CPU storage"),
        }
    }

    fn cuda_fwd(
        &self,
        s_base: &CudaStorage,
        l_base: &Layout,
        s_hidden: &CudaStorage,
        l_hidden: &Layout,
        s_b: &CudaStorage,
        l_b: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        if !l_base.is_contiguous() || !l_hidden.is_contiguous() || !l_b.is_contiguous() {
            candle_core::bail!("CudaLoraAddBf16 CUDA path requires contiguous inputs");
        }
        let out_storage = s_base.try_clone(l_base)?;
        let out_shape = Shape::from(l_base.dims().to_vec());
        let out_layout = Layout::contiguous(out_shape.clone());
        kiln_rmsnorm_kernel::lora_add_bf16_storage(
            &out_storage,
            &out_layout,
            s_base,
            l_base,
            s_hidden,
            l_hidden,
            s_b,
            l_b,
            self.scale,
        )
        .map_err(|e| candle_core::Error::Msg(format!("CudaLoraAddBf16 CUDA add: {e:?}")))?;
        Ok((out_storage, out_shape))
    }

    fn bwd(
        &self,
        _base: &Tensor,
        hidden: &Tensor,
        b: &Tensor,
        _res: &Tensor,
        grad_y: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let grad_base = to_dtype_if_needed(grad_y, DType::BF16)?;
        let rows = grad_y.dim(0)?;
        let tile_rows = cuda_lora_bwd_tile_rows().min(rows.max(1));
        let b_f32 = to_dtype_if_needed(b, DType::F32)?;
        let mut grad_hidden_tiles = Vec::with_capacity(rows.div_ceil(tile_rows));
        let mut grad_b_acc: Option<Tensor> = None;

        for start in (0..rows).step_by(tile_rows) {
            let len = (rows - start).min(tile_rows);
            let grad_y_tile = grad_y.narrow(0, start, len)?;
            let grad_y_tile_f32 = to_dtype_if_needed(&grad_y_tile, DType::F32)?;
            let hidden_tile = hidden.narrow(0, start, len)?;
            let hidden_tile_f32 = to_dtype_if_needed(&hidden_tile, DType::F32)?;
            let grad_hidden_tile = grad_y_tile_f32
                .matmul(&b_f32)?
                .affine(self.scale as f64, 0.0)?;
            let grad_b_tile = grad_y_tile_f32
                .t()?
                .matmul(&hidden_tile_f32)?
                .affine(self.scale as f64, 0.0)?;
            grad_hidden_tiles.push(grad_hidden_tile);
            grad_b_acc = Some(match grad_b_acc {
                Some(acc) => (acc + grad_b_tile)?,
                None => grad_b_tile,
            });
        }

        let grad_hidden_refs = grad_hidden_tiles.iter().collect::<Vec<_>>();
        let grad_hidden = Tensor::cat(&grad_hidden_refs, 0)?;
        let Some(grad_b_acc) = grad_b_acc else {
            candle_core::bail!("CudaLoraAddBf16 backward produced no B gradient tiles");
        };
        let grad_b = grad_b_acc.to_dtype(b.dtype())?;
        Ok((Some(grad_base), Some(grad_hidden), Some(grad_b)))
    }
}

fn linear_with_lora_t_decode_if(
    use_metal_decode_gemv: bool,
    x: &Tensor,
    weight_t: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    if use_metal_decode_gemv {
        linear_with_lora_t_decode(x, weight_t, lora, lora_scale)
    } else {
        linear_with_lora_t(x, weight_t, lora, lora_scale)
    }
}

fn linear_with_lora_t_backend_decode_if(
    backend: Option<&dyn BackendRuntime>,
    use_metal_decode_gemv: bool,
    x: &Tensor,
    weight_t: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(out) = cuda_lora_linear_training_bf16(x, weight_t, lora, lora_scale)? {
        return Ok(out);
    }
    if let Some(backend) = backend {
        // Autograd-tracked input → prefer the autograd-safe Vulkan
        // CustomOp1 (linear_prefill_apply). The existing linear_decode
        // returns a candle leaf tensor, which silently loses the
        // gradient w.r.t. `x` and produces wrong gradients to upstream
        // LoRA params. Routing by track_op() preserves the existing
        // leaf-fast path for inference (where autograd doesn't matter)
        // while routing training through the parity-tested CustomOp.
        // Gated on KILN_VULKAN_LINEAR=1 inside linear_prefill_apply
        // until production validation flips the default on.
        if x.track_op() {
            if let Some(base) = backend.linear_prefill_apply(x, weight_t)? {
                return add_lora_delta_to_base(Some(backend), base, x, lora, lora_scale);
            }
        }
        if let Some(base) = backend.linear_decode(x, weight_t)? {
            return add_lora_delta_to_base(Some(backend), base, x, lora, lora_scale);
        }
        // Last-ditch: try the autograd-safe path even for non-tracked
        // inputs in case the backend supports a shape/dtype combo
        // linear_decode declines.
        if let Some(base) = backend.linear_prefill_apply(x, weight_t)? {
            return add_lora_delta_to_base(Some(backend), base, x, lora, lora_scale);
        }
    }
    if lora.is_some() {
        let base = linear_with_lora_t_decode_if(use_metal_decode_gemv, x, weight_t, None, 0.0)?;
        return add_lora_delta_to_base(backend, base, x, lora, lora_scale);
    }
    linear_with_lora_t_decode_if(use_metal_decode_gemv, x, weight_t, None, 0.0)
}

#[cfg(feature = "metal")]
fn metal_attn_gate_debug_active() -> bool {
    crate::mtp_debug::is_subop_capture_armed()
        || crate::mtp_debug::current_b12_layer_is_31()
        || crate::mtp_debug::is_c7_sdpa_capture_armed()
        || crate::mtp_debug::is_mtp_fp32_head_armed()
        || crate::mtp_debug::is_mtp_single_token_self_attn_armed()
}

fn attention_output_gate_decode_if(
    _use_metal_decode_gemv: bool,
    attn_output: Tensor,
    gate: Option<&Tensor>,
) -> Result<Tensor> {
    let Some(gate) = gate else {
        return Ok(attn_output);
    };

    #[cfg(feature = "metal")]
    {
        if _use_metal_decode_gemv
            && !metal_attn_gate_debug_active()
            && crate::backend::metal::metal_attn_gate_sigmoid_mul_supports(&attn_output, gate)
        {
            kiln_nvtx::range!(c"kiln/attn/output_gate_fused");
            return crate::backend::metal::metal_attn_gate_sigmoid_mul_bf16(&attn_output, gate)
                .context("metal attn gate sigmoid/mul failed");
        }
    }

    #[cfg(feature = "cuda")]
    {
        if let Some(out) = cuda_sigmoid_mul_training_bf16(&attn_output, gate)? {
            return Ok(out);
        }
        if !cuda_fused_attn_sigmoid_mul_disabled()
            && !attn_output.track_op()
            && !gate.track_op()
            && kiln_rmsnorm_kernel::supports_sigmoid_mul(&attn_output, gate)
        {
            kiln_nvtx::range!(c"kiln/attn/output_gate_cuda_fused");
            return kiln_rmsnorm_kernel::fused_sigmoid_mul(&attn_output, gate)
                .context("cuda attn gate sigmoid/mul failed");
        }
    }

    let sigmoid_gate = cuda_sigmoid(gate)?;
    Ok((attn_output * sigmoid_gate)?)
}

#[cfg(feature = "cuda")]
fn cuda_sigmoid_mul_training_bf16(x: &Tensor, gate: &Tensor) -> Result<Option<Tensor>> {
    if cuda_fused_attn_sigmoid_mul_disabled()
        || (!x.track_op() && !gate.track_op())
        || !kiln_rmsnorm_kernel::supports_sigmoid_mul(x, gate)
    {
        return Ok(None);
    }
    let out = x
        .apply_op2(gate, CudaSigmoidMulTrainingBf16)
        .context("cuda training sigmoid/mul CustomOp2")?;
    Ok(Some(out))
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
struct CudaSigmoidMulTrainingBf16;

#[cfg(feature = "cuda")]
impl CustomOp2 for CudaSigmoidMulTrainingBf16 {
    fn name(&self) -> &'static str {
        "kiln-cuda-sigmoid-mul-training-bf16"
    }

    fn cpu_fwd(
        &self,
        _s_x: &CpuStorage,
        _l_x: &Layout,
        _s_gate: &CpuStorage,
        _l_gate: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("CudaSigmoidMulTrainingBf16 requires CUDA inputs");
    }

    fn cuda_fwd(
        &self,
        s_x: &CudaStorage,
        l_x: &Layout,
        s_gate: &CudaStorage,
        l_gate: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        if !l_x.is_contiguous()
            || !l_gate.is_contiguous()
            || l_x.start_offset() != 0
            || l_gate.start_offset() != 0
        {
            candle_core::bail!(
                "CudaSigmoidMulTrainingBf16 CUDA path requires compact contiguous inputs"
            );
        }
        let out_storage = s_x.try_clone(l_x)?;
        let out_shape = Shape::from(l_x.dims().to_vec());
        let out_layout = Layout::contiguous(out_shape.clone());
        kiln_rmsnorm_kernel::fused_sigmoid_mul_storage(
            &out_storage,
            &out_layout,
            s_x,
            l_x,
            s_gate,
            l_gate,
        )
        .map_err(|e| {
            candle_core::Error::Msg(format!("CudaSigmoidMulTrainingBf16 CUDA fwd: {e:?}"))
        })?;
        Ok((out_storage, out_shape))
    }

    fn bwd(
        &self,
        x: &Tensor,
        gate: &Tensor,
        _res: &Tensor,
        grad_y: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        let dims = x.dims();
        if dims != gate.dims() || dims != grad_y.dims() {
            candle_core::bail!(
                "CudaSigmoidMulTrainingBf16 backward shape mismatch x={:?} gate={:?} grad={:?}",
                dims,
                gate.dims(),
                grad_y.dims()
            );
        }
        let Some(&width) = dims.last() else {
            candle_core::bail!("CudaSigmoidMulTrainingBf16 backward requires non-scalar input");
        };
        let rows = x.elem_count() / width.max(1);
        if rows == 0 {
            return Ok((
                Some(Tensor::zeros(dims, x.dtype(), x.device())?),
                Some(Tensor::zeros(dims, gate.dtype(), gate.device())?),
            ));
        }
        let Device::Cuda(cuda_device) = x.device() else {
            candle_core::bail!("CudaSigmoidMulTrainingBf16 backward requires CUDA input");
        };
        x.device().synchronize()?;

        let x_2d = x.reshape((rows, width))?;
        let gate_2d = gate.reshape((rows, width))?;
        let grad_y_2d = grad_y.reshape((rows, width))?;
        let out_shape = Shape::from(vec![rows, width]);
        let mut grad_x_storage = unsafe { cuda_device.alloc_uninit(&out_shape, x.dtype()) }
            .map_err(|e| candle_core::Error::Msg(format!("sigmoid-mul bwd grad_x alloc: {e:?}")))?;
        let mut grad_gate_storage = unsafe { cuda_device.alloc_uninit(&out_shape, gate.dtype()) }
            .map_err(|e| {
            candle_core::Error::Msg(format!("sigmoid-mul bwd grad_gate alloc: {e:?}"))
        })?;
        let tile_rows = cuda_lora_bwd_tile_rows().min(rows.max(1));

        for start in (0..rows).step_by(tile_rows) {
            let len = (rows - start).min(tile_rows);
            let x_tile = x_2d.narrow(0, start, len)?;
            let gate_tile = gate_2d.narrow(0, start, len)?;
            let grad_y_tile = grad_y_2d.narrow(0, start, len)?;

            let sigmoid_gate = (gate_tile.neg()?.exp()? + 1.0)?.recip()?;
            let grad_x_tile = (grad_y_tile.clone() * sigmoid_gate.clone())?;
            let (grad_x_tile_storage, grad_x_tile_layout) = grad_x_tile.storage_and_layout();
            let Storage::Cuda(grad_x_tile_storage) = &*grad_x_tile_storage else {
                candle_core::bail!(
                    "CudaSigmoidMulTrainingBf16 backward produced non-CUDA grad_x tile"
                );
            };
            grad_x_tile_storage.copy2d(
                &mut grad_x_storage,
                len,
                width,
                width,
                width,
                grad_x_tile_layout.start_offset(),
                start * width,
            )?;

            let sigmoid_f32 = sigmoid_gate.to_dtype(DType::F32)?;
            let one_minus_sigmoid = (sigmoid_f32.neg()? + 1.0)?;
            let gate_deriv = (sigmoid_f32 * one_minus_sigmoid)?;
            let grad_gate_tile =
                (grad_y_tile.to_dtype(DType::F32)? * x_tile.to_dtype(DType::F32)?)?;
            let grad_gate_tile = (grad_gate_tile * gate_deriv)?.to_dtype(gate.dtype())?;
            let (grad_gate_tile_storage, grad_gate_tile_layout) =
                grad_gate_tile.storage_and_layout();
            let Storage::Cuda(grad_gate_tile_storage) = &*grad_gate_tile_storage else {
                candle_core::bail!(
                    "CudaSigmoidMulTrainingBf16 backward produced non-CUDA grad_gate tile"
                );
            };
            grad_gate_tile_storage.copy2d(
                &mut grad_gate_storage,
                len,
                width,
                width,
                width,
                grad_gate_tile_layout.start_offset(),
                start * width,
            )?;
        }

        let grad_x = Tensor::from_storage(
            Storage::Cuda(grad_x_storage),
            out_shape.clone(),
            BackpropOp::none(),
            false,
        )
        .reshape(dims)?;
        let grad_gate = Tensor::from_storage(
            Storage::Cuda(grad_gate_storage),
            out_shape,
            BackpropOp::none(),
            false,
        )
        .reshape(dims)?;
        Ok((Some(grad_x), Some(grad_gate)))
    }
}

fn full_attn_qkv_proj_decode_if(
    backend: &dyn BackendRuntime,
    use_metal_decode_gemv: bool,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    lora_layer: Option<&LoraLayerWeights>,
    lora_scale: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    #[cfg(feature = "metal")]
    {
        if use_metal_decode_gemv
            && lora_layer.is_none()
            && attn_weights.q_proj_marlin.is_none()
            && !crate::mtp_debug::is_mtp_fp32_head_armed()
            && !crate::mtp_debug::is_mtp_single_token_self_attn_armed()
            && !crate::mtp_debug::current_b12_layer_is_31()
            && crate::backend::metal::metal_fused_qkv_transposed_coop_gemv_supports(
                x,
                &attn_weights.q_proj_t,
                &attn_weights.k_proj_t,
                &attn_weights.v_proj_t,
            )
        {
            kiln_nvtx::range!(c"kiln/proj/qkv_fused");
            return crate::backend::metal::metal_fused_qkv_transposed_coop_gemv_bf16(
                x,
                &attn_weights.q_proj_t,
                &attn_weights.k_proj_t,
                &attn_weights.v_proj_t,
            )
            .context("metal fused QKV projection failed");
        }
    }

    if lora_layer.is_none()
        && attn_weights.q_proj_marlin.is_none()
        && !crate::mtp_debug::is_mtp_fp32_head_armed()
        && !crate::mtp_debug::is_mtp_single_token_self_attn_armed()
    {
        #[cfg(feature = "cuda")]
        {
            if cuda_full_attn_qkv_in_proj_enabled()
                && !x.track_op()
                && x.dtype() == DType::BF16
                && matches!(x.device(), Device::Cuda(_))
            {
                if let Some(qkv_proj_t) = attn_weights.qkv_proj_t.as_ref() {
                    if let Ok((_, seq_len, hidden)) = x.dims3() {
                        let q_dim = attn_weights.q_proj_t.dim(1)?;
                        let k_dim = attn_weights.k_proj_t.dim(1)?;
                        let v_dim = attn_weights.v_proj_t.dim(1)?;
                        if seq_len == 1
                            && qkv_proj_t.dtype() == DType::BF16
                            && !qkv_proj_t.track_op()
                            && matches!(qkv_proj_t.device(), Device::Cuda(_))
                            && qkv_proj_t.is_contiguous()
                            && qkv_proj_t.dims() == [hidden, q_dim + k_dim + v_dim]
                        {
                            let qkv = broadcast_matmul_cpu_compatible(x, qkv_proj_t)
                                .context("cuda full-attn combined Q/K/V projection matmul")?;
                            let q_raw = qkv.narrow(2, 0, q_dim)?;
                            let k_raw = qkv.narrow(2, q_dim, k_dim)?;
                            let v = qkv.narrow(2, q_dim + k_dim, v_dim)?;
                            return Ok((q_raw, k_raw, v));
                        }
                    }
                }
            }
        }
        if let Some(out) = backend.full_attn_qkv_decode(
            x,
            &attn_weights.q_proj_t,
            &attn_weights.k_proj_t,
            &attn_weights.v_proj_t,
        )? {
            kiln_nvtx::range!(c"kiln/proj/qkv_fused");
            return Ok(out);
        }
    }

    let q_raw = q_proj_forward_decode_if(
        Some(backend),
        use_metal_decode_gemv,
        x,
        attn_weights,
        lora_layer.and_then(|l| l.q_proj.as_ref()),
        lora_scale,
    )?;
    let k = linear_with_lora_t_backend_decode_if(
        Some(backend),
        use_metal_decode_gemv,
        x,
        &attn_weights.k_proj_t,
        lora_layer.and_then(|l| l.k_proj.as_ref()),
        lora_scale,
    )?;
    let v = linear_with_lora_t_backend_decode_if(
        Some(backend),
        use_metal_decode_gemv,
        x,
        &attn_weights.v_proj_t,
        lora_layer.and_then(|l| l.v_proj.as_ref()),
        lora_scale,
    )?;
    Ok((q_raw, k, v))
}

/// CUDA-compatible softmax on last dimension.
///
/// `candle_nn::ops::softmax_last_dim` lacks a CUDA kernel, so we implement it
/// manually: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`.
fn cuda_softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let max_val = x.max_keepdim(candle_core::D::Minus1)?;
    let shifted = x.broadcast_sub(&max_val)?;
    let exp_shifted = shifted.exp()?;
    let sum_exp = exp_shifted.sum_keepdim(candle_core::D::Minus1)?;
    Ok(exp_shifted.broadcast_div(&sum_exp)?)
}

/// Compute attention using a backend FlashAttention-2 fast path.
///
/// Takes Q, K, V in `[batch, seq_len, num_heads, head_dim]` layout (pre-transpose).
/// K/V may have fewer heads than Q (GQA); they are expanded to match Q's head count
/// before calling the flash kernel, which requires uniform head counts.
///
/// Routes through `backend.flash_attn_prefill`. Returns `Ok(Some(out))` with
/// `out` shaped `[batch, seq_len, num_heads * head_dim]` (already reshaped for
/// output projection) when the backend handles it, or `Ok(None)` when the
/// backend declines — callers must fall back to the portable candle path.
fn flash_attention_forward(
    backend: &dyn BackendRuntime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Option<Tensor>> {
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let causal = true;

    // GQA: the vendored CUDA FA2 wrapper receives `num_heads_k` separately, so
    // it can consume grouped K/V directly. Other backends still take the
    // historic expanded layout through this trait method.
    let (k, v) = if num_heads != num_kv_heads && backend.name() != "cuda" {
        let gqa_ratio = num_heads / num_kv_heads;
        let (batch, kv_len, _kv_heads, hd) = k.dims4()?;
        // [batch, kv_len, num_kv_heads, head_dim] -> [batch, kv_len, num_heads, head_dim]
        let k = k
            .unsqueeze(3)?
            .expand(&[batch, kv_len, num_kv_heads, gqa_ratio, hd])?
            .contiguous()?
            .reshape((batch, kv_len, num_heads, hd))?;
        let v = v
            .unsqueeze(3)?
            .expand(&[batch, kv_len, num_kv_heads, gqa_ratio, hd])?
            .contiguous()?
            .reshape((batch, kv_len, num_heads, hd))?;
        (k, v)
    } else {
        (k.clone(), v.clone())
    };

    let Some(attn_output) = backend.flash_attn_prefill(q, &k, &v, softmax_scale, causal)? else {
        return Ok(None);
    };

    // Reshape to [batch, seq_len, hidden]
    let (batch, seq_len, _heads, _hd) = attn_output.dims4()?;
    let attn_output = attn_output.reshape((batch, seq_len, num_heads * head_dim))?;
    Ok(Some(attn_output))
}

#[cfg(feature = "cuda")]
fn cuda_flash_attention_training_disabled() -> bool {
    env_truthy("KILN_DISABLE_CUDA_FLASH_ATTN_TRAINING")
}

#[cfg(feature = "cuda")]
fn cuda_flash_attention_training_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Option<Tensor>> {
    if cuda_flash_attention_training_disabled() || !any_tensor_tracks_op(&[q, k, v]) {
        return Ok(None);
    }
    if q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || !matches!(q.device(), Device::Cuda(_))
        || !matches!(k.device(), Device::Cuda(_))
        || !matches!(v.device(), Device::Cuda(_))
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
        || !matches!(head_dim, 128 | 256)
        || num_kv_heads == 0
        || num_heads % num_kv_heads != 0
    {
        return Ok(None);
    }
    let (bq, _sq, hq, dq) = q.dims4()?;
    let (bk, sk, hk, dk) = k.dims4()?;
    let (bv, sv, hv, dv) = v.dims4()?;
    if bq != bk
        || bq != bv
        || sk != sv
        || hq != num_heads
        || hk != num_kv_heads
        || hv != num_kv_heads
        || dq != head_dim
        || dk != head_dim
        || dv != head_dim
    {
        return Ok(None);
    }

    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let out = q
        .apply_op3(
            k,
            v,
            CudaFlashAttentionTrainingBf16 {
                softmax_scale,
                causal: true,
            },
        )
        .context("cuda training FlashAttention CustomOp3")?;
    Ok(Some(out))
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
struct CudaFlashAttentionTrainingBf16 {
    softmax_scale: f32,
    causal: bool,
}

#[cfg(feature = "cuda")]
impl CustomOp3 for CudaFlashAttentionTrainingBf16 {
    fn name(&self) -> &'static str {
        "kiln-cuda-flash-attn-training-bf16"
    }

    fn cpu_fwd(
        &self,
        _s_q: &CpuStorage,
        _l_q: &Layout,
        _s_k: &CpuStorage,
        _l_k: &Layout,
        _s_v: &CpuStorage,
        _l_v: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("CudaFlashAttentionTrainingBf16 requires CUDA inputs");
    }

    fn cuda_fwd(
        &self,
        s_q: &CudaStorage,
        l_q: &Layout,
        s_k: &CudaStorage,
        l_k: &Layout,
        s_v: &CudaStorage,
        l_v: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        if !l_q.is_contiguous()
            || !l_k.is_contiguous()
            || !l_v.is_contiguous()
            || l_q.start_offset() != 0
            || l_k.start_offset() != 0
            || l_v.start_offset() != 0
        {
            candle_core::bail!(
                "CudaFlashAttentionTrainingBf16 CUDA path requires compact contiguous inputs"
            );
        }
        let q = Tensor::from_storage(
            Storage::Cuda(s_q.try_clone(l_q)?),
            Shape::from(l_q.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let k = Tensor::from_storage(
            Storage::Cuda(s_k.try_clone(l_k)?),
            Shape::from(l_k.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let v = Tensor::from_storage(
            Storage::Cuda(s_v.try_clone(l_v)?),
            Shape::from(l_v.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let (out, _softmax_lse) =
            kiln_flash_attn::flash_attn_fwd(&q, &k, &v, self.softmax_scale, self.causal).map_err(
                |e| {
                    candle_core::Error::Msg(format!(
                        "CudaFlashAttentionTrainingBf16 CUDA fwd: {e:?}"
                    ))
                },
            )?;
        let out_shape = Shape::from(out.dims().to_vec());
        let (storage, layout) = out.storage_and_layout();
        let storage = storage.try_clone(layout)?;
        match storage {
            Storage::Cuda(storage) => Ok((storage, out_shape)),
            _ => candle_core::bail!("CudaFlashAttentionTrainingBf16 produced non-CUDA storage"),
        }
    }

    fn bwd(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        res: &Tensor,
        grad_y: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let (recomputed_out, softmax_lse) =
            kiln_flash_attn::flash_attn_fwd(q, k, v, self.softmax_scale, self.causal).map_err(
                |e| {
                    candle_core::Error::Msg(format!(
                        "CudaFlashAttentionTrainingBf16 bwd recompute: {e:?}"
                    ))
                },
            )?;
        drop(recomputed_out);
        let dout = if grad_y.dtype() == DType::BF16 {
            grad_y.clone()
        } else {
            grad_y.to_dtype(DType::BF16)?
        };
        let (dq, dk, dv) = kiln_flash_attn::flash_attn_bwd(
            &dout,
            q,
            k,
            v,
            res,
            &softmax_lse,
            self.softmax_scale,
            self.causal,
        )
        .map_err(|e| {
            candle_core::Error::Msg(format!("CudaFlashAttentionTrainingBf16 bwd: {e:?}"))
        })?;
        Ok((Some(dq), Some(dk), Some(dv)))
    }
}

/// Compute attention using a backend fast path when Q/K/V are already in
/// `[batch, heads, seq_len, head_dim]` layout.
///
/// The paged prefill path transposes Q/K/V to this layout before writing the
/// KV cache. Metal's fused SDPA also consumes this layout, so this variant
/// avoids transposing all three tensors back to token-major only for the
/// backend to transpose them again.
fn flash_attention_forward_head_major(
    backend: &dyn BackendRuntime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<Option<Tensor>> {
    if !backend.supports_flash_attn_prefill_head_major() {
        return Ok(None);
    }

    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let causal = true;

    let Some(attn_output) =
        backend.flash_attn_prefill_head_major(q, k, v, softmax_scale, causal)?
    else {
        return Ok(None);
    };

    let (batch, _heads, seq_len, _hd) = attn_output.dims4()?;
    let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
        batch,
        seq_len,
        num_heads * head_dim,
    ))?;
    Ok(Some(attn_output))
}

/// GPU-ready tensors organized by layer, converted from raw `ModelWeights` bytes.
pub struct GpuWeights {
    /// Token embedding table: [vocab_size, hidden_size]
    pub embed_tokens: Tensor,
    /// Pre-transposed token embedding table for tied LM head: [hidden_size, vocab_size], contiguous.
    /// Computed once at load to avoid re-transposing the ~778 MiB bf16 matrix on every decode step
    /// (was 48% of ucopy_bf16 / ~43% of GPU time per PR #113 profile).
    pub embed_tokens_t: Tensor,
    /// Per-layer weights
    pub layers: Vec<GpuLayerWeights>,
    /// Final RMSNorm weight: [hidden_size]
    pub final_norm: Tensor,
    /// Cached rotary inv_freq tensor, shape `[half_rotary]`, F32 on device.
    /// Computed once at load time from `config.rotary_dim()` and `config.rope_theta`
    /// so the RoPE hot path can reuse it instead of rebuilding a fresh `Vec<f32>` +
    /// HtoD upload on every layer's attention call (~8 × per token in prefill).
    pub rotary_inv_freq: Tensor,
    /// Native MTP (Multi-Token Prediction) head tensors, when the checkpoint
    /// shipped them and the loader surfaced a `ModelWeights.mtp`.
    ///
    /// The slot is lazy by default so desktop startup does not upload the MTP
    /// tensors to Metal unless a request actually resolves to native MTP.
    /// `None` still means the checkpoint does not support native MTP.
    pub mtp: Option<MtpGpuWeightsSlot>,
}

/// GPU-ready native MTP head tensors.
///
/// Mirrors [`crate::weights::MtpWeights`] after upload. The `lm_head` is tied
/// to the base model's token embedding, so this struct intentionally does NOT
/// carry its own `lm_head` tensor — the spec-decode forward pass reuses
/// [`GpuWeights::embed_tokens_t`] for the final projection.
///
/// The inner [`GpuLayerWeights`] is re-used for the MTP transformer layer so
/// the forward pass can dispatch through the same full-attention kernels
/// (q/k/v/o_proj, q_norm, k_norm, input/post_attention_layernorm, SwiGLU MLP)
/// that it uses for the base model's eight full-attention layers. The loader
/// already rejects any MTP checkpoint that resolves as linear attention, so
/// the inner `attention` field is always `GpuAttentionWeights::Full(_)`.
pub struct MtpGpuWeights {
    /// Concat-then-project: `[hidden_size, 2 * hidden_size]`, BF16 on device.
    /// Ingests `concat(norm_embed, norm_hidden)` → produces `[seq, hidden_size]`.
    pub fc: Tensor,
    /// Cached `fc` transpose for the forward hot path: `[2 * hidden_size, hidden_size]`,
    /// materialized contiguously once at load time.
    /// Same transpose-caching pattern as the base model's `*_proj_t` fields
    /// (PRs #117/#124/#128) — eliminates a per-draft-step `.t().contiguous()`
    /// on a 26 MiB bf16 matrix when drafting.
    pub fc_t: Tensor,
    /// RMSNorm weight for the draft-candidate's token embedding. `[hidden_size]`.
    pub pre_fc_norm_embedding: Tensor,
    /// RMSNorm weight for the base model's last hidden state. `[hidden_size]`.
    pub pre_fc_norm_hidden: Tensor,
    /// Single MTP transformer layer. The loader validates this is always a
    /// full-attention layer, so `layer.attention` is `Full(...)` at runtime.
    pub layer: GpuLayerWeights,
    /// Final RMSNorm weight before the tied lm_head. `[hidden_size]`.
    pub final_layernorm: Tensor,
}

/// Lazy GPU materialization for native MTP tensors.
///
/// Routing only needs to know whether MTP exists; the first actual MTP forward
/// pays the upload cost. This avoids blocking macOS desktop readiness on an
/// MTP path that the server uses only for short greedy prompts.
pub struct MtpGpuWeightsSlot {
    weights: OnceLock<MtpGpuWeights>,
    source: Option<MtpGpuSource>,
    device: Device,
    init_lock: Mutex<()>,
}

#[derive(Clone)]
enum MtpGpuSource {
    Loaded(MtpWeights),
    Deferred(DeferredMtpSource),
}

impl MtpGpuWeightsSlot {
    pub fn lazy(source: MtpWeights, device: &Device) -> Self {
        Self {
            weights: OnceLock::new(),
            source: Some(MtpGpuSource::Loaded(source)),
            device: device.clone(),
            init_lock: Mutex::new(()),
        }
    }

    pub fn lazy_deferred(source: DeferredMtpSource, device: &Device) -> Self {
        Self {
            weights: OnceLock::new(),
            source: Some(MtpGpuSource::Deferred(source)),
            device: device.clone(),
            init_lock: Mutex::new(()),
        }
    }

    pub fn eager(weights: MtpGpuWeights, device: &Device) -> Self {
        let slot = Self {
            weights: OnceLock::new(),
            source: None,
            device: device.clone(),
            init_lock: Mutex::new(()),
        };
        let _ = slot.weights.set(weights);
        slot
    }

    pub fn is_uploaded(&self) -> bool {
        self.weights.get().is_some()
    }

    pub fn get_or_upload(&self) -> Result<&MtpGpuWeights> {
        if let Some(weights) = self.weights.get() {
            return Ok(weights);
        }

        let _guard = self
            .init_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock MTP GPU upload slot: {e}"))?;
        if let Some(weights) = self.weights.get() {
            return Ok(weights);
        }

        let source = self
            .source
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("native MTP GPU slot is empty and has no CPU source"))?;
        let mtp_weights = match source {
            MtpGpuSource::Loaded(weights) => weights.clone(),
            MtpGpuSource::Deferred(source) => {
                let load_start = std::time::Instant::now();
                let loaded = crate::loader::load_deferred_mtp(source)
                    .context("deferred native MTP CPU load")?;
                tracing::info!(
                    load_elapsed_ms = load_start.elapsed().as_millis() as u64,
                    "deferred native MTP CPU load complete"
                );
                loaded
            }
        };
        let projection_load_cache =
            ProjectionLoadCache::new(&self.device).context("mtp projection load cache")?;
        let upload_start = std::time::Instant::now();
        let uploaded = upload_mtp_gpu_weights(&mtp_weights, &self.device, &projection_load_cache)
            .context("lazy native MTP GPU upload")?;
        let upload_elapsed_ms = upload_start.elapsed().as_millis();
        self.weights
            .set(uploaded)
            .map_err(|_| anyhow::anyhow!("native MTP GPU weights were initialized twice"))?;
        tracing::info!(
            upload_elapsed_ms = upload_elapsed_ms as u64,
            "lazy native MTP GPU upload complete"
        );

        self.weights
            .get()
            .ok_or_else(|| anyhow::anyhow!("native MTP GPU upload completed but slot is empty"))
    }
}

fn upload_mtp_gpu_weights(
    mtp_w: &MtpWeights,
    device: &Device,
    projection_load_cache: &ProjectionLoadCache,
) -> Result<MtpGpuWeights> {
    let (fc, fc_t) = projection_tensors_for_load(&mtp_w.fc, device, projection_load_cache)
        .context("mtp.fc projection tensors")?;
    let pre_fc_norm_embedding = weight_to_tensor(&mtp_w.pre_fc_norm_embedding, device)
        .context("mtp.pre_fc_norm_embedding")?;
    let pre_fc_norm_hidden =
        weight_to_tensor(&mtp_w.pre_fc_norm_hidden, device).context("mtp.pre_fc_norm_hidden")?;
    let final_layernorm =
        weight_to_tensor(&mtp_w.final_layernorm, device).context("mtp.final_layernorm")?;

    // The MTP inner transformer layer. Loader guarantees this is a
    // full-attention layer (bails otherwise). Keep the upload local to MTP
    // rather than adding it to Marlin packing; native MTP uses one layer and
    // is not on the long-prompt desktop route.
    let mtp_layer = {
        let lw = &mtp_w.layer;
        let ctx = |name: &str| format!("mtp.layer {name}");

        let input_layernorm =
            weight_to_tensor(&lw.input_layernorm, device).context(ctx("input_layernorm"))?;
        let post_attention_layernorm = weight_to_tensor(&lw.post_attention_layernorm, device)
            .context(ctx("post_attention_layernorm"))?;

        let attention = match &lw.attention {
            crate::weights::AttentionWeights::Full(attn) => {
                let attn_proj = projection_tensors_for_load_batch(
                    &[
                        ("q_proj", &attn.q_proj),
                        ("k_proj", &attn.k_proj),
                        ("v_proj", &attn.v_proj),
                        ("o_proj", &attn.o_proj),
                    ],
                    device,
                    projection_load_cache,
                )
                .context(ctx("attention projection tensors"))?;
                let mut attn_proj = attn_proj.into_iter();
                let (q_proj, q_proj_t) = attn_proj.next().context(ctx("q_proj missing"))?;
                let (k_proj, k_proj_t) = attn_proj.next().context(ctx("k_proj missing"))?;
                let (v_proj, v_proj_t) = attn_proj.next().context(ctx("v_proj missing"))?;
                let (o_proj, o_proj_t) = attn_proj.next().context(ctx("o_proj missing"))?;
                GpuAttentionWeights::Full(GpuFullAttentionWeights {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm: weight_to_tensor(&attn.q_norm, device).context(ctx("q_norm"))?,
                    k_norm: weight_to_tensor(&attn.k_norm, device).context(ctx("k_norm"))?,
                    q_proj_t,
                    k_proj_t,
                    v_proj_t,
                    qkv_proj_t: None,
                    o_proj_t,
                    q_proj_marlin: None,
                })
            }
            crate::weights::AttentionWeights::Linear(_) => {
                anyhow::bail!(
                    "MTP layer resolved as linear attention - loader should have caught this"
                );
            }
        };

        let mlp_proj = projection_tensors_for_load_batch(
            &[
                ("gate_proj", &lw.mlp.gate_proj),
                ("up_proj", &lw.mlp.up_proj),
                ("down_proj", &lw.mlp.down_proj),
            ],
            device,
            projection_load_cache,
        )
        .context(ctx("mlp projection tensors"))?;
        let mut mlp_proj = mlp_proj.into_iter();
        let (gate_proj, gate_proj_t) = mlp_proj.next().context(ctx("gate_proj missing"))?;
        let (up_proj, up_proj_t) = mlp_proj.next().context(ctx("up_proj missing"))?;
        let (down_proj, down_proj_t) = mlp_proj.next().context(ctx("down_proj missing"))?;

        GpuLayerWeights {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp: GpuFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
            },
        }
    };

    Ok(MtpGpuWeights {
        fc,
        fc_t,
        pre_fc_norm_embedding,
        pre_fc_norm_hidden,
        layer: mtp_layer,
        final_layernorm,
    })
}

/// Compute the rotary-embedding `inv_freq` tensor once and upload it to `device`.
///
/// `inv_freq_i = 1.0 / (rope_theta ^ (2i / rotary_dim))` for `i` in `0..rotary_dim/2`.
/// The result is an F32 tensor of shape `[rotary_dim / 2]`.
pub fn compute_rotary_inv_freq(
    rotary_dim: usize,
    rope_theta: f64,
    device: &Device,
) -> Result<Tensor> {
    let half_rotary = rotary_dim / 2;
    let inv_freq: Vec<f32> = (0..half_rotary)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / rotary_dim as f64) as f32)
        .collect();
    let t = Tensor::new(inv_freq.as_slice(), device)
        .context("failed to build rotary inv_freq tensor")?;
    Ok(t)
}

/// One transformer layer's tensors on device.
pub struct GpuLayerWeights {
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
    pub attention: GpuAttentionWeights,
    pub mlp: GpuFfnWeights,
}

/// Attention weights on device.
pub enum GpuAttentionWeights {
    Full(GpuFullAttentionWeights),
    Linear(GpuLinearAttentionWeights),
}

pub struct GpuFullAttentionWeights {
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    /// Cached q_proj transpose for the forward hot path, materialized
    /// contiguously once at load time.
    /// Avoids re-transposing bf16 projection weights on every layer / every step.
    /// Per PR #124 PROFILING.md: attention projection ucopy_bf16 was ~6.9% of decode GPU time.
    pub q_proj_t: Tensor,
    pub k_proj_t: Tensor,
    pub v_proj_t: Tensor,
    /// Optional cached `[hidden, q_raw + k + v]` transpose for CUDA decode.
    /// This combines the full-attention Q/K/V projections into one matmul on
    /// forward-only single-token fast paths without disturbing the separate
    /// transposes used by training, LoRA, Marlin, and debug captures.
    pub qkv_proj_t: Option<Tensor>,
    pub o_proj_t: Tensor,
    /// Optional Marlin W4A16-packed q_proj. Populated at load time when the
    /// `KILN_W4A16=1` env var is set on a CUDA build whose q_proj shape fits
    /// Marlin's tile constraints (k%128 && n%256). When present, the forward
    /// path routes q_proj through the Marlin kernel instead of the BF16
    /// `broadcast_matmul` via `q_proj_t`. LoRA deltas are still applied on top.
    pub q_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
}

pub struct GpuLinearAttentionWeights {
    pub in_proj_qkv: Tensor,
    pub in_proj_z: Tensor,
    pub out_proj: Tensor,
    pub in_proj_a: Tensor,
    pub in_proj_b: Tensor,
    pub conv1d: Tensor,
    pub norm: Tensor,
    pub a_log: Tensor,
    pub a_log_gates: Tensor,
    pub dt_bias: Tensor,
    /// Cached GDN projection transposes for the forward hot path,
    /// materialized contiguously once at load time.
    /// Same fix class as PR #128 (MLP/full-attn pre-transpose) and PR #117 (embed_tokens_t).
    /// Per Phase 6 PROFILING.md: GDN in_proj+out_proj together accounted for ~95% of
    /// decode-time `ucopy_bf16` mass on Qwen3.5-4B; eliminating the per-step `.t()` copies
    /// removes that bandwidth completely.
    pub in_proj_qkv_t: Tensor,
    pub in_proj_z_t: Tensor,
    pub in_proj_a_t: Tensor,
    pub in_proj_b_t: Tensor,
    /// Optional cached `[hidden, 2 * nv]` transpose that combines the small
    /// prefill/decode A/B projections into one matmul on backend fast paths.
    pub in_proj_ab_t: Option<Tensor>,
    pub out_proj_t: Tensor,
    /// Optional Marlin W4A16-packed GDN out_proj. Populated at load time
    /// when `KILN_W4A16_GDN_OUT_PROJ=1` is set on a CUDA build whose
    /// out_proj shape fits Marlin's tile constraints (`k%128 && n%256`).
    /// When present, the GDN forward path uses Marlin for the projection
    /// instead of `broadcast_matmul` via `out_proj_t`. This is gated
    /// behind a separate opt-in from the existing `KILN_W4A16` because the
    /// GDN out_proj is the last linear layer in the GDN block before the
    /// residual add — int4 quantization there is more sensitive to
    /// quality drift than the in-projections or the MLP, so deployments
    /// opt in only after their own quality A/B passes.
    pub out_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
}

pub struct GpuFfnWeights {
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
    /// Cached MLP projection transposes for the forward hot path,
    /// materialized contiguously once at load time.
    /// Avoids re-transposing bf16 projection weights on every layer / every step.
    /// Per PR #124 PROFILING.md: MLP projection ucopy_bf16 was 50.7% of decode GPU time
    /// (61.8% of all ucopy_bf16 mass). Same class of fix as PR #117 (embed_tokens_t).
    pub gate_proj_t: Tensor,
    pub up_proj_t: Tensor,
    pub down_proj_t: Tensor,
    /// Optional Marlin W4A16-packed MLP projections. Populated at load time
    /// when the `KILN_W4A16=1` env var is set on a CUDA build whose projection
    /// shape fits Marlin's tile constraints (k%128 && n%256). When present,
    /// the forward path routes the corresponding projection through the
    /// Marlin kernel instead of the BF16 `broadcast_matmul` via `*_t`. LoRA
    /// deltas are still applied on top. Mirrors the q_proj_marlin wire-in
    /// from PR #149 but expands coverage from 8 layers (q_proj on full-attn
    /// layers only) to all 32 layers × 3 MLP projections.
    pub gate_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
    pub up_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
    pub down_proj_marlin: Option<crate::marlin_proj::MarlinPackedProj>,
}

/// State for Gated DeltaNet linear attention layers.
///
/// Each linear attention layer maintains:
/// - A recurrent state matrix S of shape `[batch, num_value_heads, key_head_dim, value_head_dim]`
/// - A conv1d sliding window buffer of shape `[batch, conv_dim, kernel_size - 1]`
///
/// This state is O(1) in sequence length — it does not grow with the number of tokens processed.
pub struct LinearAttentionState {
    /// Per-layer recurrent state S. Length = number of linear attention layers.
    pub recurrent_states: Vec<Tensor>,
    /// Per-layer conv1d sliding window buffers. Length = number of linear attention layers.
    pub conv_states: Vec<Tensor>,
}

impl LinearAttentionState {
    /// Create fresh zero-initialized state for all linear attention layers.
    pub fn new(config: &kiln_core::config::ModelConfig, device: &Device) -> Result<Self> {
        Self::new_with_batch_and_recurrent_dtype(
            config,
            1,
            device,
            Self::training_recurrent_dtype(config, device),
        )
    }

    /// Create fresh inference state for all linear attention layers.
    ///
    /// CUDA/Metal inference and explicitly named Vulkan inference use the same
    /// dtype as the model weights so decode does not cast every GDN recurrent
    /// state into and back out of the hot kernel dtype on every token. `new`
    /// keeps the training/test default.
    pub fn new_for_inference(
        config: &kiln_core::config::ModelConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_batch_for_inference(config, 1, device)
    }

    /// Create fresh inference state for `batch` independent decode rows.
    pub fn new_with_batch_for_inference(
        config: &kiln_core::config::ModelConfig,
        batch: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_batch_for_inference_backend(config, batch, device, None)
    }

    /// Create fresh inference state for `batch` decode rows, allowing callers
    /// whose accelerator is not represented by Candle's `Device` enum to name
    /// the backend explicitly.
    pub fn new_with_batch_for_inference_backend(
        config: &kiln_core::config::ModelConfig,
        batch: usize,
        device: &Device,
        backend_name: Option<&str>,
    ) -> Result<Self> {
        Self::new_with_batch_and_recurrent_dtype(
            config,
            batch,
            device,
            Self::inference_recurrent_dtype(config, device, backend_name),
        )
    }

    /// Create fresh zero-initialized state for all linear attention layers and
    /// `batch` independent decode rows.
    pub fn new_with_batch(
        config: &kiln_core::config::ModelConfig,
        batch: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_batch_and_recurrent_dtype(
            config,
            batch,
            device,
            Self::training_recurrent_dtype(config, device),
        )
    }

    fn training_recurrent_dtype(config: &kiln_core::config::ModelConfig, device: &Device) -> DType {
        match (device, config.dtype) {
            (Device::Metal(_), kiln_core::config::DType::BF16) => DType::BF16,
            (Device::Metal(_), kiln_core::config::DType::FP16) => DType::F16,
            _ => DType::F32,
        }
    }

    fn inference_recurrent_dtype(
        config: &kiln_core::config::ModelConfig,
        device: &Device,
        backend_name: Option<&str>,
    ) -> DType {
        let cuda_bf16_state_disabled =
            std::env::var("KILN_DISABLE_CUDA_BF16_INFERENCE_STATE").is_ok();
        let vulkan_bf16_state_disabled =
            std::env::var("KILN_DISABLE_VULKAN_BF16_INFERENCE_STATE").is_ok();
        match (device, config.dtype) {
            (Device::Cuda(_), _) if cuda_bf16_state_disabled => {
                Self::training_recurrent_dtype(config, device)
            }
            (Device::Cuda(_), kiln_core::config::DType::BF16)
            | (Device::Metal(_), kiln_core::config::DType::BF16) => DType::BF16,
            (Device::Cuda(_), kiln_core::config::DType::FP16)
            | (Device::Metal(_), kiln_core::config::DType::FP16) => DType::F16,
            (_, kiln_core::config::DType::BF16)
                if backend_name == Some("vulkan") && !vulkan_bf16_state_disabled =>
            {
                DType::BF16
            }
            (_, kiln_core::config::DType::FP16)
                if backend_name == Some("vulkan") && !vulkan_bf16_state_disabled =>
            {
                DType::F16
            }
            _ => DType::F32,
        }
    }

    fn new_with_batch_and_recurrent_dtype(
        config: &kiln_core::config::ModelConfig,
        batch: usize,
        device: &Device,
        recurrent_dtype: DType,
    ) -> Result<Self> {
        anyhow::ensure!(batch > 0, "LinearAttentionState batch must be positive");
        let num_linear_layers = config.num_layers - config.num_full_attention_layers;
        let nv = config.linear_num_value_heads;
        let dk = config.linear_key_head_dim;
        let dv = config.linear_value_head_dim;
        let conv_dim = config.linear_qkv_dim();
        let k_minus_1 = config.linear_conv_kernel_dim.saturating_sub(1);

        let mut recurrent_states = Vec::with_capacity(num_linear_layers);
        let mut conv_states = Vec::with_capacity(num_linear_layers);

        for _ in 0..num_linear_layers {
            recurrent_states.push(Tensor::zeros((batch, nv, dk, dv), recurrent_dtype, device)?);
            conv_states.push(Tensor::zeros(
                (batch, conv_dim, k_minus_1),
                DType::F32,
                device,
            )?);
        }

        Ok(Self {
            recurrent_states,
            conv_states,
        })
    }

    /// Return the shared batch dimension across all recurrent and conv states.
    pub fn batch_size(&self) -> Result<usize> {
        if self.recurrent_states.len() != self.conv_states.len() {
            anyhow::bail!(
                "LinearAttentionState batch_size: recurrent/conv layer count mismatch ({} vs {})",
                self.recurrent_states.len(),
                self.conv_states.len()
            );
        }

        let first = self
            .recurrent_states
            .first()
            .context("LinearAttentionState batch_size: no recurrent states")?;
        let batch = first.dim(0)?;
        for (idx, tensor) in self.recurrent_states.iter().enumerate() {
            anyhow::ensure!(
                tensor.dim(0)? == batch,
                "LinearAttentionState batch_size: recurrent state {idx} batch mismatch"
            );
        }
        for (idx, tensor) in self.conv_states.iter().enumerate() {
            anyhow::ensure!(
                tensor.dim(0)? == batch,
                "LinearAttentionState batch_size: conv state {idx} batch mismatch"
            );
        }
        Ok(batch)
    }

    /// Assemble a batched GDN state from one-row per-request states.
    pub fn from_batch_rows(rows: &[&Self]) -> Result<Self> {
        anyhow::ensure!(
            !rows.is_empty(),
            "LinearAttentionState::from_batch_rows requires at least one row"
        );
        let num_layers = rows[0].recurrent_states.len();
        anyhow::ensure!(
            rows[0].conv_states.len() == num_layers,
            "LinearAttentionState::from_batch_rows row 0 recurrent/conv layer count mismatch"
        );

        for (idx, row) in rows.iter().enumerate() {
            anyhow::ensure!(
                row.recurrent_states.len() == num_layers && row.conv_states.len() == num_layers,
                "LinearAttentionState::from_batch_rows row {idx} layer count mismatch"
            );
            let row_batch = row.batch_size()?;
            anyhow::ensure!(
                row_batch == 1,
                "LinearAttentionState::from_batch_rows row {idx} has batch size {}, expected 1",
                row_batch
            );
        }

        // Defensive dtype normalization: rows must share a dtype for `Tensor::cat`.
        // The canonical recurrent dtype is whatever `new_with_batch` produced for
        // row 0 (F32 on CUDA, BF16/F16 on Metal). If any other row drifted (e.g. a
        // prior decode error left state mid-conversion in BF16), cast it back to
        // row 0's dtype so cat succeeds. Same for conv state.
        let mut recurrent_states = Vec::with_capacity(num_layers);
        let mut conv_states = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let target_recurrent_dtype = rows[0].recurrent_states[layer_idx].dtype();
            let mut recurrent_owned: Vec<Tensor> = Vec::with_capacity(rows.len());
            for (row_idx, row) in rows.iter().enumerate() {
                let t = &row.recurrent_states[layer_idx];
                if t.dtype() != target_recurrent_dtype {
                    tracing::debug!(
                        layer = layer_idx,
                        row = row_idx,
                        from = ?t.dtype(),
                        to = ?target_recurrent_dtype,
                        "from_batch_rows: normalizing recurrent state dtype before cat"
                    );
                    recurrent_owned.push(t.to_dtype(target_recurrent_dtype)?);
                } else {
                    recurrent_owned.push(t.clone());
                }
            }
            let recurrent_refs: Vec<&Tensor> = recurrent_owned.iter().collect();

            let target_conv_dtype = rows[0].conv_states[layer_idx].dtype();
            let mut conv_owned: Vec<Tensor> = Vec::with_capacity(rows.len());
            for (row_idx, row) in rows.iter().enumerate() {
                let t = &row.conv_states[layer_idx];
                if t.dtype() != target_conv_dtype {
                    tracing::debug!(
                        layer = layer_idx,
                        row = row_idx,
                        from = ?t.dtype(),
                        to = ?target_conv_dtype,
                        "from_batch_rows: normalizing conv state dtype before cat"
                    );
                    conv_owned.push(t.to_dtype(target_conv_dtype)?);
                } else {
                    conv_owned.push(t.clone());
                }
            }
            let conv_refs: Vec<&Tensor> = conv_owned.iter().collect();

            // `Tensor::cat` already produces a contiguous output tensor, so
            // the trailing `.contiguous()` was a no-op that nevertheless
            // re-checked the layout on every cat — and on the hot decode
            // path that meant one redundant CPU-side check per (layer, step,
            // state-kind) tuple = 24 GDN layers × 2 states × steps. Removing
            // it shaves a small amount of dispatch overhead off the
            // `batch_state_assemble` stage. The runtime path is identical
            // because cat is the only source feeding these tensors.
            recurrent_states.push(Tensor::cat(&recurrent_refs, 0)?);
            conv_states.push(Tensor::cat(&conv_refs, 0)?);
        }

        Ok(Self {
            recurrent_states,
            conv_states,
        })
    }

    /// Split a batched state into one-row states in batch order.
    pub fn split_batch_rows(&self) -> Result<Vec<Self>> {
        let batch = self.batch_size()?;
        let mut rows = Vec::with_capacity(batch);
        for batch_idx in 0..batch {
            let mut recurrent_states = Vec::with_capacity(self.recurrent_states.len());
            let mut conv_states = Vec::with_capacity(self.conv_states.len());
            for tensor in &self.recurrent_states {
                recurrent_states.push(tensor.narrow(0, batch_idx, 1)?.contiguous()?);
            }
            for tensor in &self.conv_states {
                conv_states.push(tensor.narrow(0, batch_idx, 1)?.contiguous()?);
            }
            rows.push(Self {
                recurrent_states,
                conv_states,
            });
        }
        Ok(rows)
    }

    /// Overwrite one-row destination states from the rows of this batched state.
    pub fn scatter_batch_rows(&self, destinations: &mut [&mut Self]) -> Result<()> {
        let rows = self.split_batch_rows()?;
        anyhow::ensure!(
            destinations.len() == rows.len(),
            "LinearAttentionState::scatter_batch_rows destination count mismatch ({} vs {})",
            destinations.len(),
            rows.len()
        );
        for (dst, row) in destinations.iter_mut().zip(rows.iter()) {
            dst.restore_from(row)?;
        }
        Ok(())
    }

    /// Replace one-row destination tensors from this batched state.
    ///
    /// This avoids the extra `restore_from` copies in [`Self::scatter_batch_rows`]
    /// for scheduler-owned batch decode rows, where CUDA graph pointer stability
    /// is not required because batch-size > 1 graph replay is not used.
    pub fn scatter_batch_rows_replace(&self, destinations: &mut [&mut Self]) -> Result<()> {
        let batch = self.batch_size()?;
        anyhow::ensure!(
            destinations.len() == batch,
            "LinearAttentionState::scatter_batch_rows_replace destination count mismatch ({} vs {})",
            destinations.len(),
            batch
        );

        for (row_idx, dst) in destinations.iter_mut().enumerate() {
            anyhow::ensure!(
                dst.recurrent_states.len() == self.recurrent_states.len(),
                "LinearAttentionState::scatter_batch_rows_replace recurrent layer count mismatch for row {row_idx} ({} vs {})",
                dst.recurrent_states.len(),
                self.recurrent_states.len()
            );
            anyhow::ensure!(
                dst.conv_states.len() == self.conv_states.len(),
                "LinearAttentionState::scatter_batch_rows_replace conv layer count mismatch for row {row_idx} ({} vs {})",
                dst.conv_states.len(),
                self.conv_states.len()
            );

            for (dst_tensor, src_tensor) in dst
                .recurrent_states
                .iter_mut()
                .zip(self.recurrent_states.iter())
            {
                *dst_tensor = src_tensor.narrow(0, row_idx, 1)?.contiguous()?;
            }
            for (dst_tensor, src_tensor) in dst.conv_states.iter_mut().zip(self.conv_states.iter())
            {
                *dst_tensor = src_tensor.narrow(0, row_idx, 1)?.contiguous()?;
            }
        }

        Ok(())
    }

    /// Assemble backend-resident recurrent row buffers into this batched state.
    ///
    /// The CPU tensors still carry the same shapes/dtypes as the portable
    /// path, but a backend may bind device-resident state buffers to their
    /// tensor IDs so the decode recurrent kernel can avoid re-uploading stale
    /// CPU state.
    pub fn assemble_gdn_recurrent_resident_batch_rows(
        &self,
        backend: &dyn BackendRuntime,
        rows: &[&Self],
    ) -> Result<bool> {
        let batch = self.batch_size()?;
        anyhow::ensure!(
            rows.len() == batch,
            "LinearAttentionState::assemble_gdn_recurrent_resident_batch_rows row count mismatch ({} vs {})",
            rows.len(),
            batch
        );
        let mut assembled_any = false;
        for layer_idx in 0..self.recurrent_states.len() {
            let row_tensors: Vec<&Tensor> = rows
                .iter()
                .map(|row| &row.recurrent_states[layer_idx])
                .collect();
            assembled_any |= backend.assemble_gdn_recurrent_resident_batch_rows(
                &row_tensors,
                &self.recurrent_states[layer_idx],
            )?;
        }
        Ok(assembled_any)
    }

    /// Refresh THIS batched state's recurrent + conv tensors *in place*
    /// from the supplied per-row states, preserving device pointers.
    /// Required by the multi-batch CUDA graph replay path: the captured
    /// graph holds the persistent slot's device addresses, so refreshing
    /// must not replace the tensors.
    ///
    /// Uses [`Tensor::slice_set`] per row + per layer, which writes the
    /// source bytes into the destination's existing storage. After this
    /// call, `self.recurrent_states[layer_idx][row]` byte-matches
    /// `rows[row].recurrent_states[layer_idx]`, same for `conv_states`.
    ///
    /// The inverse direction (persistent → per-row, e.g. after a graph
    /// replay) still uses [`Self::scatter_batch_rows_replace_with_backend`]
    /// which is allowed to replace per-row tensors — only the batched
    /// slot's pointers must stay pinned.
    pub fn refresh_batched_state_from_rows_in_place(
        &mut self,
        rows: &[&Self],
    ) -> Result<()> {
        let batch = self.batch_size()?;
        anyhow::ensure!(
            rows.len() == batch,
            "refresh_batched_state_from_rows_in_place row count mismatch ({} vs {})",
            rows.len(),
            batch
        );
        anyhow::ensure!(
            rows.iter()
                .all(|r| r.recurrent_states.len() == self.recurrent_states.len()
                    && r.conv_states.len() == self.conv_states.len()),
            "refresh_batched_state_from_rows_in_place: per-row state layer-count mismatch"
        );
        for layer_idx in 0..self.recurrent_states.len() {
            for (row_idx, src) in rows.iter().enumerate() {
                self.recurrent_states[layer_idx]
                    .slice_set(&src.recurrent_states[layer_idx], 0, row_idx)
                    .with_context(|| {
                        format!(
                            "refresh recurrent state row {row_idx} into persistent batched slot at layer {layer_idx}"
                        )
                    })?;
            }
        }
        for layer_idx in 0..self.conv_states.len() {
            for (row_idx, src) in rows.iter().enumerate() {
                self.conv_states[layer_idx]
                    .slice_set(&src.conv_states[layer_idx], 0, row_idx)
                    .with_context(|| {
                        format!(
                            "refresh conv state row {row_idx} into persistent batched slot at layer {layer_idx}"
                        )
                    })?;
            }
        }
        Ok(())
    }

    /// Replace one-row destination tensors, preserving backend-resident
    /// recurrent state when the backend owns a fresher batch buffer.
    pub fn scatter_batch_rows_replace_with_backend(
        &self,
        backend: &dyn BackendRuntime,
        destinations: &mut [&mut Self],
    ) -> Result<()> {
        let batch = self.batch_size()?;
        anyhow::ensure!(
            destinations.len() == batch,
            "LinearAttentionState::scatter_batch_rows_replace_with_backend destination count mismatch ({} vs {})",
            destinations.len(),
            batch
        );

        for (row_idx, dst) in destinations.iter_mut().enumerate() {
            anyhow::ensure!(
                dst.recurrent_states.len() == self.recurrent_states.len(),
                "LinearAttentionState::scatter_batch_rows_replace_with_backend recurrent layer count mismatch for row {row_idx} ({} vs {})",
                dst.recurrent_states.len(),
                self.recurrent_states.len()
            );
            anyhow::ensure!(
                dst.conv_states.len() == self.conv_states.len(),
                "LinearAttentionState::scatter_batch_rows_replace_with_backend conv layer count mismatch for row {row_idx} ({} vs {})",
                dst.conv_states.len(),
                self.conv_states.len()
            );
        }

        for layer_idx in 0..self.recurrent_states.len() {
            let mut dst_tensors: Vec<&mut Tensor> = destinations
                .iter_mut()
                .map(|dst| &mut dst.recurrent_states[layer_idx])
                .collect();
            if !backend.scatter_gdn_recurrent_resident_batch_rows(
                &self.recurrent_states[layer_idx],
                &mut dst_tensors,
            )? {
                for (row_idx, dst_tensor) in dst_tensors.into_iter().enumerate() {
                    *dst_tensor = self.recurrent_states[layer_idx]
                        .narrow(0, row_idx, 1)?
                        .contiguous()?;
                }
            }
        }

        for layer_idx in 0..self.conv_states.len() {
            for (row_idx, dst) in destinations.iter_mut().enumerate() {
                dst.conv_states[layer_idx] = self.conv_states[layer_idx]
                    .narrow(0, row_idx, 1)?
                    .contiguous()?;
            }
        }

        Ok(())
    }

    pub fn materialize_gdn_recurrent_resident_states(
        &mut self,
        backend: &dyn BackendRuntime,
    ) -> Result<()> {
        for state in &mut self.recurrent_states {
            backend.materialize_gdn_recurrent_resident_state(state)?;
        }
        Ok(())
    }

    pub fn evict_gdn_recurrent_resident_states(&self, backend: &dyn BackendRuntime) {
        for state in &self.recurrent_states {
            backend.evict_gdn_recurrent_resident_state(state);
        }
    }

    pub fn has_any_gdn_recurrent_resident_state(&self, backend: &dyn BackendRuntime) -> bool {
        self.recurrent_states
            .iter()
            .any(|state| backend.has_gdn_recurrent_resident_state(state))
    }

    pub fn has_all_gdn_recurrent_resident_states(&self, backend: &dyn BackendRuntime) -> bool {
        !self.recurrent_states.is_empty()
            && self
                .recurrent_states
                .iter()
                .all(|state| backend.has_gdn_recurrent_resident_state(state))
    }

    /// Capture the current GDN recurrent + conv state into a fresh shadow
    /// `LinearAttentionState`. Used by speculative decoding to preserve the
    /// base model's O(1) GDN state before advancing into a draft: if any
    /// proposed token is rejected, [`Self::restore_from`] puts it back.
    ///
    /// This snapshot allocates new device tensors and issues a
    /// `cudaMemcpyDeviceToDevice` per layer. For Qwen3.5-4B that is
    /// 24 × (recurrent ≈ 2 MiB + conv ≈ 24 KiB) ≈ 49 MiB per snapshot, which
    /// is acceptable for WIP scaffolding. The follow-up PR replaces this with
    /// the ping-pong shadow-slot pattern from the existing KV-cache draft
    /// code path (no per-step alloc, two pre-allocated slots swapped via
    /// index) to bring overhead to zero.
    pub fn snapshot(&self) -> Result<Self> {
        let mut recurrent_states = Vec::with_capacity(self.recurrent_states.len());
        for t in &self.recurrent_states {
            recurrent_states.push(t.copy().context("snapshot recurrent state")?);
        }
        let mut conv_states = Vec::with_capacity(self.conv_states.len());
        for t in &self.conv_states {
            conv_states.push(t.copy().context("snapshot conv state")?);
        }
        Ok(Self {
            recurrent_states,
            conv_states,
        })
    }

    /// Snapshot for decode rollback.
    ///
    /// Recurrent tensors are replaced on update, so Arc-cloning their handles
    /// preserves the pre-step value without a device copy. Conv state is mutated
    /// in-place by the Metal/CUDA update kernels, so it must still be copied.
    pub fn snapshot_for_decode_rollback(&self) -> Result<Self> {
        self.snapshot_for_decode_rollback_prefix(self.recurrent_states.len())
    }

    /// Snapshot only the linear-attention prefix needed by a draft model.
    ///
    /// Skip-layer drafting runs `model_forward_segment(..., 0, draft_layers)`,
    /// so it never touches GDN states after that layer prefix. Carrying all
    /// 24 Qwen3.5-4B GDN states in the draft snapshot wastes device copies on
    /// every speculative step.
    pub fn snapshot_for_decode_rollback_prefix(&self, num_linear_layers: usize) -> Result<Self> {
        if num_linear_layers > self.recurrent_states.len() {
            anyhow::bail!(
                "LinearAttentionState::snapshot_for_decode_rollback_prefix: requested {} recurrent states, only {} available",
                num_linear_layers,
                self.recurrent_states.len()
            );
        }
        if num_linear_layers > self.conv_states.len() {
            anyhow::bail!(
                "LinearAttentionState::snapshot_for_decode_rollback_prefix: requested {} conv states, only {} available",
                num_linear_layers,
                self.conv_states.len()
            );
        }
        let recurrent_states = self.recurrent_states[..num_linear_layers].to_vec();
        let mut conv_states = Vec::with_capacity(num_linear_layers);
        for t in &self.conv_states[..num_linear_layers] {
            conv_states.push(t.copy().context("snapshot conv state")?);
        }
        Ok(Self {
            recurrent_states,
            conv_states,
        })
    }

    /// Restore this state from a previously captured [`Self::snapshot`].
    ///
    /// Checks that the shapes/counts match — a mismatch indicates the caller
    /// mixed up snapshots from different sessions, which would be a logic bug
    /// in the spec-decode loop. Overwrites the current tensors in place so
    /// downstream GPU pointers (e.g. those captured inside a CUDA graph) stay
    /// valid. The follow-up ping-pong rewrite folds this into a zero-copy
    /// slot swap; this correctness-first copy implementation is the scaffold.
    pub fn restore_from(&mut self, snapshot: &Self) -> Result<()> {
        if self.recurrent_states.len() != snapshot.recurrent_states.len() {
            anyhow::bail!(
                "LinearAttentionState::restore_from: recurrent_states len mismatch ({} vs {})",
                self.recurrent_states.len(),
                snapshot.recurrent_states.len()
            );
        }
        if self.conv_states.len() != snapshot.conv_states.len() {
            anyhow::bail!(
                "LinearAttentionState::restore_from: conv_states len mismatch ({} vs {})",
                self.conv_states.len(),
                snapshot.conv_states.len()
            );
        }
        for (dst, src) in self
            .recurrent_states
            .iter_mut()
            .zip(snapshot.recurrent_states.iter())
        {
            *dst = src.copy().context("restore recurrent state")?;
        }
        for (dst, src) in self.conv_states.iter_mut().zip(snapshot.conv_states.iter()) {
            *dst = src.copy().context("restore conv state")?;
        }
        Ok(())
    }

    /// Restore from [`Self::snapshot_for_decode_rollback`] without recopying
    /// recurrent state. The snapshot owns fresh conv-state copies, so assigning
    /// their tensor handles is enough to restore the old conv buffers as well.
    pub fn restore_from_decode_rollback(&mut self, snapshot: &Self) -> Result<()> {
        if self.recurrent_states.len() != snapshot.recurrent_states.len() {
            anyhow::bail!(
                "LinearAttentionState::restore_from_decode_rollback: recurrent_states len mismatch ({} vs {})",
                self.recurrent_states.len(),
                snapshot.recurrent_states.len()
            );
        }
        if self.conv_states.len() != snapshot.conv_states.len() {
            anyhow::bail!(
                "LinearAttentionState::restore_from_decode_rollback: conv_states len mismatch ({} vs {})",
                self.conv_states.len(),
                snapshot.conv_states.len()
            );
        }
        self.recurrent_states.clone_from(&snapshot.recurrent_states);
        self.conv_states.clone_from(&snapshot.conv_states);
        Ok(())
    }
}

/// Convert a `WeightTensor` (raw bytes + shape + dtype) to a candle `Tensor` on `device`.
fn weight_to_tensor(w: &WeightTensor, device: &Device) -> Result<Tensor> {
    let dtype = weight_dtype(w);
    let t = Tensor::from_raw_buffer(w.as_bytes(), dtype, &w.shape, device)
        .context("failed to create tensor from raw buffer")?;
    Ok(t)
}

fn weight_dtype(w: &WeightTensor) -> DType {
    match w.dtype {
        TensorDType::F16 => DType::F16,
        TensorDType::BF16 => DType::BF16,
        TensorDType::F32 => DType::F32,
    }
}

const TRANSPOSE_ROW_TILE: usize = 32;
const TRANSPOSE_COL_TILE: usize = 32;
const PARALLEL_TRANSPOSE_MIN_BYTES: usize = 1 << 20;
const PARALLEL_TRANSPOSE_ROW_CHUNK: usize = 64;

#[inline(always)]
fn copy_transpose_elem_unaligned<T: Copy>(data: &[u8], out: &mut [u8], src: usize, dst: usize) {
    // Safetensors byte offsets are not guaranteed to satisfy Rust alignment
    // for typed views, so use unaligned loads/stores while still avoiding a
    // tiny `memmove` call per BF16/F32 element.
    unsafe {
        let value = std::ptr::read_unaligned(data.as_ptr().add(src).cast::<T>());
        std::ptr::write_unaligned(out.as_mut_ptr().add(dst).cast::<T>(), value);
    }
}

fn transpose_weight_bytes_typed<T: Copy + Send + Sync>(
    data: &[u8],
    out: &mut [u8],
    rows: usize,
    cols: usize,
) {
    let elem_size = std::mem::size_of::<T>();

    if data.len() < PARALLEL_TRANSPOSE_MIN_BYTES {
        for row0 in (0..rows).step_by(TRANSPOSE_ROW_TILE) {
            let row_end = (row0 + TRANSPOSE_ROW_TILE).min(rows);
            for col0 in (0..cols).step_by(TRANSPOSE_COL_TILE) {
                let col_end = (col0 + TRANSPOSE_COL_TILE).min(cols);
                for row in row0..row_end {
                    for col in col0..col_end {
                        let src = (row * cols + col) * elem_size;
                        let dst = (col * rows + row) * elem_size;
                        copy_transpose_elem_unaligned::<T>(data, out, src, dst);
                    }
                }
            }
        }
    } else {
        transpose_weight_bytes_typed_parallel_rows::<T>(data, out, rows, cols);
    }
}

fn transpose_weight_bytes_typed_parallel_rows<T: Copy + Send + Sync>(
    data: &[u8],
    out: &mut [u8],
    rows: usize,
    cols: usize,
) {
    use rayon::prelude::*;

    let elem_size = std::mem::size_of::<T>();
    let out_col_stride = rows * elem_size;
    let chunks = rows.div_ceil(PARALLEL_TRANSPOSE_ROW_CHUNK);
    let out_addr = out.as_mut_ptr() as usize;

    (0..chunks).into_par_iter().for_each(|chunk_idx| {
        let row0 = chunk_idx * PARALLEL_TRANSPOSE_ROW_CHUNK;
        let row_end = (row0 + PARALLEL_TRANSPOSE_ROW_CHUNK).min(rows);
        let out_ptr = out_addr as *mut u8;

        for row in row0..row_end {
            let mut src = row * cols * elem_size;
            let mut dst = row * elem_size;
            for _ in 0..cols {
                // SAFETY: row chunks are disjoint. For any source element
                // `(row, col)`, the transposed destination is `(col, row)`,
                // so different row chunks write non-overlapping bytes within
                // each output column. `transposed_weight_bytes_2d` validated
                // data/out lengths before dispatching here.
                unsafe {
                    let value = std::ptr::read_unaligned(data.as_ptr().add(src).cast::<T>());
                    std::ptr::write_unaligned(out_ptr.add(dst).cast::<T>(), value);
                }
                src += elem_size;
                dst += out_col_stride;
            }
        }
    });
}

fn transpose_weight_bytes_generic(
    data: &[u8],
    out: &mut [u8],
    rows: usize,
    cols: usize,
    elem_size: usize,
) {
    if data.len() < PARALLEL_TRANSPOSE_MIN_BYTES {
        for row0 in (0..rows).step_by(TRANSPOSE_ROW_TILE) {
            let row_end = (row0 + TRANSPOSE_ROW_TILE).min(rows);
            for col0 in (0..cols).step_by(TRANSPOSE_COL_TILE) {
                let col_end = (col0 + TRANSPOSE_COL_TILE).min(cols);
                for row in row0..row_end {
                    for col in col0..col_end {
                        let src = (row * cols + col) * elem_size;
                        let dst = (col * rows + row) * elem_size;
                        out[dst..dst + elem_size].copy_from_slice(&data[src..src + elem_size]);
                    }
                }
            }
        }
    } else {
        use rayon::prelude::*;

        let out_col_stride = rows * elem_size;
        let out_block_stride = out_col_stride * TRANSPOSE_COL_TILE;
        out.par_chunks_mut(out_block_stride)
            .enumerate()
            .for_each(|(block_idx, out_block)| {
                let col0 = block_idx * TRANSPOSE_COL_TILE;
                let col_end = (col0 + (out_block.len() / out_col_stride)).min(cols);
                for row0 in (0..rows).step_by(TRANSPOSE_ROW_TILE) {
                    let row_end = (row0 + TRANSPOSE_ROW_TILE).min(rows);
                    for col in col0..col_end {
                        let out_col = col - col0;
                        let out_base = out_col * out_col_stride;
                        for row in row0..row_end {
                            let src = (row * cols + col) * elem_size;
                            let dst = out_base + row * elem_size;
                            out_block[dst..dst + elem_size]
                                .copy_from_slice(&data[src..src + elem_size]);
                        }
                    }
                }
            });
    }
}

pub(crate) fn transposed_weight_bytes_2d(w: &WeightTensor) -> Result<(Vec<u8>, [usize; 2])> {
    anyhow::ensure!(
        w.shape.len() == 2,
        "direct transposed weight upload requires a rank-2 tensor, got shape {:?}",
        w.shape
    );
    let rows = w.shape[0];
    let cols = w.shape[1];
    let elem_size = w.dtype.size_bytes();
    let data = w.as_bytes();
    let expected_len = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(elem_size))
        .context("weight tensor byte size overflow")?;
    anyhow::ensure!(
        data.len() == expected_len,
        "weight tensor data length mismatch: got {} bytes, expected {} bytes for shape {:?} and dtype {}",
        data.len(),
        expected_len,
        w.shape,
        w.dtype
    );

    let mut out = vec![0u8; data.len()];
    match elem_size {
        1 => transpose_weight_bytes_typed::<u8>(data, &mut out, rows, cols),
        2 => transpose_weight_bytes_typed::<u16>(data, &mut out, rows, cols),
        4 => transpose_weight_bytes_typed::<u32>(data, &mut out, rows, cols),
        8 => transpose_weight_bytes_typed::<u64>(data, &mut out, rows, cols),
        _ => transpose_weight_bytes_generic(data, &mut out, rows, cols, elem_size),
    }

    Ok((out, [cols, rows]))
}

fn weight_to_transposed_tensor_2d(w: &WeightTensor, device: &Device) -> Result<Tensor> {
    let data = transposed_weight_bytes_2d_cached_bytes(w)?;
    Tensor::from_raw_buffer(data.as_bytes(), weight_dtype(w), &data.shape(), device)
        .context("failed to create transposed tensor from raw buffer")
}

fn cached_transpose_for_weight(
    w: &WeightTensor,
    materialized: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    if matches!(device, Device::Metal(_)) {
        weight_to_transposed_tensor_2d(w, device)
    } else {
        cached_transpose(materialized)
    }
}

fn dropped_weight_stub(w: &WeightTensor, device: &Device) -> Result<Tensor> {
    Ok(Tensor::zeros((1usize,), weight_dtype(w), device)?)
}

/// True when `from_model_weights` should stub the candle CPU storage
/// for the raw `embed_tokens` table after uploading the transposed
/// view. Fires on Metal (always) and on Vulkan-active processes
/// (where the candle "device" reports as Cpu but the real compute
/// runs on a `vk::Device` that already keeps its own buffer copy of
/// every weight). On a unified-memory APU this halves the
/// embedding-table footprint by removing the duplicate candle CPU
/// mirror.
fn stub_embed_tokens_after_upload(device: &Device) -> bool {
    matches!(device, Device::Metal(_)) || crate::backend::vulkan_active()
}

#[derive(Clone)]
struct ProjectionLoadCache {
    drop_projection_originals: bool,
    drop_projection_transposes: bool,
    bf16_stub: Option<Tensor>,
    f16_stub: Option<Tensor>,
    f32_stub: Option<Tensor>,
}

impl ProjectionLoadCache {
    fn new(device: &Device) -> Result<Self> {
        let drop_projection_originals = projection_original_drop_enabled_for_device(device);
        let drop_projection_transposes =
            !drop_projection_originals && drop_projection_transposes_enabled();
        if drop_projection_originals || drop_projection_transposes {
            Ok(Self {
                drop_projection_originals,
                drop_projection_transposes,
                bf16_stub: Some(Tensor::zeros((1usize,), DType::BF16, device)?),
                f16_stub: Some(Tensor::zeros((1usize,), DType::F16, device)?),
                f32_stub: Some(Tensor::zeros((1usize,), DType::F32, device)?),
            })
        } else {
            Ok(Self {
                drop_projection_originals,
                drop_projection_transposes,
                bf16_stub: None,
                f16_stub: None,
                f32_stub: None,
            })
        }
    }

    fn stub_for(&self, dtype: DType) -> Option<Tensor> {
        match dtype {
            DType::BF16 => self.bf16_stub.clone(),
            DType::F16 => self.f16_stub.clone(),
            DType::F32 => self.f32_stub.clone(),
            _ => None,
        }
    }

    fn drops_projection_originals(&self) -> bool {
        self.drop_projection_originals
    }

    fn drops_projection_transposes(&self) -> bool {
        self.drop_projection_transposes
    }
}

fn env_enabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && !matches!(v.as_str(), "0" | "false" | "no")
        })
        .unwrap_or(false)
}

fn drop_projection_originals_enabled() -> bool {
    matches!(
        std::env::var("KILN_DROP_PROJECTION_ORIGINALS")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

fn keep_projection_originals_enabled() -> bool {
    // vk-native training reads projection weight bytes via candle's
    // CPU storage path (`extract_tensor_packed_bf16_bytes_pub`). If the
    // originals were stubbed to shape [1] during loading, those reads
    // come back as zeros and the bf16w matmul silently outputs zero —
    // the model then collapses to "embedding + residuals" and the loss
    // is bit-identical across epochs because LoRA gradients vanish.
    // Keep originals automatically whenever the process has selected the
    // Vulkan backend. Training may be submitted later, after weights are
    // loaded, so this cannot depend only on KILN_VK_NATIVE_TRAINING.
    if crate::backend::vulkan_active() || env_enabled("KILN_VK_NATIVE_TRAINING") {
        return true;
    }
    matches!(
        std::env::var("KILN_KEEP_PROJECTION_ORIGINALS")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

fn drop_projection_transposes_enabled() -> bool {
    // Only drop projection transposes when training is actually engaged
    // (KILN_VK_NATIVE_TRAINING set explicitly). Earlier this was widened to
    // every Vulkan-active process — but inference reads `in_proj_qkv_t` /
    // `in_proj_z_t` / etc. directly via `backend.linear_prefill_apply`, and
    // when those transposes are replaced with the shape-[1] BF16 stub at
    // load time the GDN prefill matmul fails with "only 2d matrixes are
    // supported [1, T, hidden] [1]" on every chat completion. Keeping the
    // originals on vulkan_active stays in keep_projection_originals_enabled
    // (it's cheap-ish and lets training start later); the transposes are
    // only dropped when the trainer is genuinely going to be the only
    // consumer.
    env_enabled("KILN_VK_NATIVE_TRAINING") && !env_enabled("KILN_KEEP_PROJECTION_TRANSPOSES")
}

fn projection_original_drop_enabled_for_device(device: &Device) -> bool {
    !keep_projection_originals_enabled()
        && (matches!(device, Device::Metal(_) | Device::Cuda(_))
            || crate::backend::vulkan_active()
            || drop_projection_originals_enabled())
}

fn projection_tensors_for_load(
    w: &WeightTensor,
    device: &Device,
    cache: &ProjectionLoadCache,
) -> Result<(Tensor, Tensor)> {
    if cache.drops_projection_originals() {
        let transposed = weight_to_transposed_tensor_2d(w, device)?;
        let original_stub = match cache.stub_for(weight_dtype(w)) {
            Some(stub) => stub,
            None => dropped_weight_stub(w, device)?,
        };
        Ok((original_stub, transposed))
    } else if cache.drops_projection_transposes() {
        let materialized = weight_to_tensor(w, device)?;
        let transposed_stub = match cache.stub_for(weight_dtype(w)) {
            Some(stub) => stub,
            None => dropped_weight_stub(w, device)?,
        };
        Ok((materialized, transposed_stub))
    } else {
        let materialized = weight_to_tensor(w, device)?;
        let transposed = cached_transpose(&materialized)?;
        Ok((materialized, transposed))
    }
}

fn parallel_projection_load_disabled() -> bool {
    matches!(
        std::env::var("KILN_DISABLE_PARALLEL_PROJECTION_LOAD")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

fn projection_tensors_for_load_batch(
    weights: &[(&str, &WeightTensor)],
    device: &Device,
    cache: &ProjectionLoadCache,
) -> Result<Vec<(Tensor, Tensor)>> {
    if !matches!(device, Device::Metal(_)) || parallel_projection_load_disabled() {
        return weights
            .iter()
            .map(|(name, w)| {
                projection_tensors_for_load(w, device, cache)
                    .with_context(|| format!("{name} projection tensors"))
            })
            .collect();
    }

    use rayon::prelude::*;

    let transposed: Result<Vec<CachedTransposedWeightBytes>> = weights
        .par_iter()
        .map(|(name, w)| {
            transposed_weight_bytes_2d_cached_bytes(w)
                .with_context(|| format!("{name} transposed projection bytes"))
        })
        .collect();

    let cache = cache.clone();
    let device = device.clone();
    transposed?
        .into_par_iter()
        .zip(weights.par_iter())
        .map(|(data, (name, w))| {
            let transposed =
                Tensor::from_raw_buffer(data.as_bytes(), weight_dtype(w), &data.shape(), &device)
                    .with_context(|| format!("{name} transposed projection upload"))?;
            let original_stub = match cache.stub_for(weight_dtype(w)) {
                Some(stub) => stub,
                None => dropped_weight_stub(w, &device)
                    .with_context(|| format!("{name} projection stub"))?,
            };
            Ok((original_stub, transposed))
        })
        .collect()
}

fn parallel_aux_load_disabled() -> bool {
    matches!(
        std::env::var("KILN_DISABLE_PARALLEL_AUX_LOAD")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

fn aux_tensors_for_load_batch(
    weights: &[(&str, &WeightTensor)],
    device: &Device,
) -> Result<Vec<Tensor>> {
    if !matches!(device, Device::Metal(_)) || parallel_aux_load_disabled() {
        return weights
            .iter()
            .map(|(name, w)| weight_to_tensor(w, device).with_context(|| format!("{name} tensor")))
            .collect();
    }

    use rayon::prelude::*;

    let device = device.clone();
    weights
        .par_iter()
        .map(|(name, w)| weight_to_tensor(w, &device).with_context(|| format!("{name} tensor")))
        .collect()
}

/// Cache a transpose for repeated GEMMs.
///
/// Matmuls on the hot path repeatedly consume these tensors, so materialize
/// the transpose once at load time instead of relying on backend-specific
/// strided access behaviour.
fn cached_transpose(weight: &Tensor) -> Result<Tensor> {
    Ok(weight.t()?.contiguous()?)
}

fn cpu_needs_f32_matmul(lhs: &Tensor, rhs: &Tensor) -> bool {
    matches!(lhs.device(), Device::Cpu) && (lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32)
}

fn broadcast_matmul_cpu_compatible(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if cpu_needs_f32_matmul(lhs, rhs) {
        let lhs_f32 = lhs.to_dtype(DType::F32)?;
        let rhs_f32 = rhs.to_dtype(DType::F32)?;
        return matmul_no_broadcast_copy(&lhs_f32, &rhs_f32);
    }
    matmul_no_broadcast_copy(lhs, rhs)
}

/// `lhs.broadcast_matmul(rhs)` for the `[B, T, K] @ [K, N] -> [B, T, N]` case
/// that drives every projection in the decoder, without paying for candle's
/// `broadcast_matmul` of materializing the broadcasted RHS via
/// `rhs.broadcast_as(...).contiguous()`. nsys (NVTX `kiln/gdn/in_proj` range)
/// showed that contiguous copy as 78 % of total GPU time at bs=4 on the
/// CUDA + GDN path — the 168 MB weight tensor was being copied across the
/// batch dim before every matmul, dwarfing the matmul itself. Flattening
/// `lhs` to 2D + `matmul(rhs)` + reshape uses the same compute path with no
/// implicit copy.
fn matmul_no_broadcast_copy(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let l_dims = lhs.dims();
    let r_dims = rhs.dims();
    if r_dims.len() == 2 && l_dims.len() >= 2 && lhs.is_contiguous() {
        let k = l_dims[l_dims.len() - 1];
        if r_dims[0] == k {
            let out_n = r_dims[1];
            let lead: usize = l_dims[..l_dims.len() - 1].iter().product();
            let lhs2d = lhs.reshape((lead, k))?;
            let out2d = lhs2d.matmul(rhs)?;
            let mut out_shape: Vec<usize> = l_dims[..l_dims.len() - 1].to_vec();
            out_shape.push(out_n);
            return Ok(out2d.reshape(out_shape)?);
        }
    }
    Ok(lhs.broadcast_matmul(rhs)?)
}

/// Vulkan-routed `[B, T, H] @ [H, D] -> [B, T, D]` matmul with autograd
/// support, falling back to [`broadcast_matmul_cpu_compatible`] when
/// the backend declines.
///
/// Phase 2 sub-step 2: GDN linear-attention layers' in_proj_qkv,
/// in_proj_z, in_proj_a, in_proj_b matmuls were going through
/// `broadcast_matmul_cpu_compatible` directly, bypassing the existing
/// Vulkan routing in `linear_with_lora_t_backend_decode_if`. This
/// helper threads them through `backend.linear_prefill_apply` (the
/// autograd-safe `CustomOp1`) when `KILN_VULKAN_LINEAR=1` is set.
/// On Qwen3.5-4B that's 24 GDN layers × 4 in-proj matmuls per layer
/// — the dominant CPU compute in training before this commit.
fn gdn_in_proj_matmul(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    weight_t: &Tensor,
) -> Result<Tensor> {
    if let Some(out) = backend.linear_prefill_apply(x, weight_t)? {
        return Ok(out);
    }
    broadcast_matmul_cpu_compatible(x, weight_t)
}

fn promote_cpu_activation(t: Tensor) -> Result<Tensor> {
    if matches!(t.device(), Device::Cpu) && t.dtype() != DType::F32 {
        Ok(t.to_dtype(DType::F32)?)
    } else {
        Ok(t)
    }
}

/// Tiny BF16 placeholder that replaces a projection's pre-transposed
/// contiguous copy (`*_proj_t`) once Marlin has absorbed it. Dropping the
/// original `Tensor` field releases the underlying CUDA buffer (the
/// refcounted `Arc<Storage>` hits zero), reclaiming the per-layer BF16
/// residency. The struct layout is preserved so every existing construction
/// site (tests, loaders) continues to compile unchanged.
fn dropped_bf16_stub(device: &Device) -> Result<Tensor> {
    Ok(Tensor::zeros((1usize,), DType::BF16, device)?)
}

/// Kill switch for the Marlin BF16 residency cleanup. Setting
/// `KILN_DISABLE_MARLIN_BF16_DROP=1` keeps the full-size `*_proj_t`
/// contiguous copies resident alongside the packed Marlin weights so the
/// previous behaviour can be reproduced for A/B measurements or parity
/// debugging. Any unset value leaves the drop enabled.
fn marlin_bf16_drop_disabled() -> bool {
    matches!(
        std::env::var("KILN_DISABLE_MARLIN_BF16_DROP")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

// ---------------------------------------------------------------------------
// Phase 7: streaming/tiled GDN prefill — env-derived configuration.
//
// Dispatch can be forced on/off via `KILN_STREAMING_PREFILL=1|0`. Without an
// override, CUDA and Metal enable streaming for long prompts where tiled
// prefill materially reduces peak activation memory. When enabled, prefill is
// performed as a sequence of fixed-size tiles (default 8192 tokens) so the
// per-layer materialized GDN intermediates only ever cover one tile at a time.
// The recurrent state in `LinearAttentionState` already provides the O(1)
// hand-off required for bit-exact agreement with the monolithic path.
// ---------------------------------------------------------------------------

/// Default tile size for streaming prefill, in tokens. Must be a multiple of
/// `GDN_CHUNK_SIZE` (64) so the chunkwise kernel never sees a partial tail
/// chunk from a tile boundary.
pub const STREAMING_PREFILL_DEFAULT_TILE: usize = 8192;
pub const STREAMING_PREFILL_CUDA_DEFAULT_THRESHOLD: usize = 8192;
pub const STREAMING_PREFILL_METAL_DEFAULT_TILE: usize = 2048;
pub const STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD: usize = 2048;
const PAGED_KV_HEAD_MAJOR_READ_MIN_TOKENS: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamingPrefillDeviceKind {
    Cpu,
    Cuda,
    Metal,
}

fn streaming_prefill_device_kind(device: &Device) -> StreamingPrefillDeviceKind {
    match device {
        Device::Cuda(_) => StreamingPrefillDeviceKind::Cuda,
        Device::Metal(_) => StreamingPrefillDeviceKind::Metal,
        _ => StreamingPrefillDeviceKind::Cpu,
    }
}

fn streaming_prefill_env_override() -> Option<bool> {
    std::env::var("KILN_STREAMING_PREFILL")
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .and_then(|v| match v.as_str() {
            "1" | "true" | "yes" => Some(true),
            "0" | "false" | "no" => Some(false),
            _ => None,
        })
}

/// Read `KILN_STREAMING_PREFILL` and return whether the streaming prefill
/// dispatch was explicitly enabled. Defaults to false for compatibility with
/// tests and non-device-aware callers.
pub fn streaming_prefill_enabled() -> bool {
    streaming_prefill_env_override().unwrap_or(false)
}

fn streaming_prefill_default_for(kind: StreamingPrefillDeviceKind, seq_len: usize) -> bool {
    match kind {
        StreamingPrefillDeviceKind::Cuda => seq_len >= STREAMING_PREFILL_CUDA_DEFAULT_THRESHOLD,
        StreamingPrefillDeviceKind::Metal => seq_len >= streaming_prefill_threshold_tokens(),
        StreamingPrefillDeviceKind::Cpu => false,
    }
}

/// Device-aware streaming prefill policy for production prefill dispatch.
///
/// Env overrides win. Without an override, long CUDA prompts use tiled prefill
/// by default because it cuts peak GDN activation memory enough to make
/// production-shaped prefill fit with workers=2 on 48 GiB GPUs; long Metal prompts use the macOS desktop
/// threshold because it improves TTFT at common chat context sizes.
pub fn streaming_prefill_enabled_for(device: &Device, seq_len: usize) -> bool {
    if let Some(enabled) = streaming_prefill_env_override() {
        return enabled;
    }
    streaming_prefill_default_for(streaming_prefill_device_kind(device), seq_len)
}

fn streaming_prefill_threshold_tokens_env_override() -> Option<usize> {
    std::env::var("KILN_STREAMING_PREFILL_THRESHOLD_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
}

/// Read `KILN_STREAMING_PREFILL_THRESHOLD_TOKENS` for Metal's automatic
/// streaming dispatch threshold. Malformed or zero values fall back to the
/// production default.
pub fn streaming_prefill_threshold_tokens() -> usize {
    streaming_prefill_threshold_tokens_env_override()
        .unwrap_or(STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD)
}

fn streaming_tile_tokens_env_override() -> Option<usize> {
    std::env::var("KILN_STREAMING_TILE_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0 && n % GDN_CHUNK_SIZE == 0)
}

/// Read `KILN_STREAMING_TILE_TOKENS` (positive multiple of `GDN_CHUNK_SIZE`).
/// Falls back to `STREAMING_PREFILL_DEFAULT_TILE` when unset, malformed, zero,
/// or not a multiple of 64.
pub fn streaming_tile_tokens() -> usize {
    streaming_tile_tokens_env_override().unwrap_or(STREAMING_PREFILL_DEFAULT_TILE)
}

/// Device-aware tile-size default. Env overrides win; otherwise Metal uses a
/// smaller tile because it measured faster for long desktop TTFT.
pub fn streaming_tile_tokens_for(device: &Device) -> usize {
    streaming_tile_tokens_env_override().unwrap_or_else(|| {
        if matches!(device, Device::Metal(_)) {
            STREAMING_PREFILL_METAL_DEFAULT_TILE
        } else {
            STREAMING_PREFILL_DEFAULT_TILE
        }
    })
}

/// Read `KILN_STREAMING_LAST_TOKEN_LM_HEAD`. Defaults to true: in streaming
/// mode only the final token's logits are needed for sampling, so the LM head
/// projection is collapsed to a single row per prefill. Set to `0` to compute
/// full per-tile logits (still throwing them away for non-final tiles, but
/// useful for parity tests against the monolithic path).
pub fn streaming_last_token_lm_head() -> bool {
    match std::env::var("KILN_STREAMING_LAST_TOKEN_LM_HEAD")
        .ok()
        .as_deref()
    {
        Some(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no"),
        None => true,
    }
}

/// Sidecar record: which slot in `layers[layer_idx]` a queued Marlin pack
/// job belongs to. Populated inline with `pack_from_bf16_batch`'s input vec
/// during the per-layer build loop, then replayed after the batch pack
/// finishes so the packed `MarlinPackedProj` lands in the right field.
#[derive(Clone, Copy, Debug)]
enum MarlinPackKind {
    QProj,
    GateProj,
    UpProj,
    DownProj,
    GdnOutProj,
}

#[derive(Debug)]
struct MarlinPackEntry {
    layer_idx: usize,
    kind: MarlinPackKind,
}

/// Install a successfully packed projection into its target layer slot,
/// and drop the corresponding pre-transposed BF16 copy unless
/// `KILN_DISABLE_MARLIN_BF16_DROP=1` has preserved it.
fn install_marlin_packed(
    layer: &mut GpuLayerWeights,
    kind: MarlinPackKind,
    packed: crate::marlin_proj::MarlinPackedProj,
    device: &Device,
    drop_disabled: bool,
) -> Result<()> {
    match kind {
        MarlinPackKind::QProj => {
            if let GpuAttentionWeights::Full(ref mut full) = layer.attention {
                full.q_proj_marlin = Some(packed);
                if !drop_disabled {
                    full.q_proj_t = dropped_bf16_stub(device)?;
                }
            }
        }
        MarlinPackKind::GateProj => {
            layer.mlp.gate_proj_marlin = Some(packed);
            if !drop_disabled {
                layer.mlp.gate_proj_t = dropped_bf16_stub(device)?;
            }
        }
        MarlinPackKind::UpProj => {
            layer.mlp.up_proj_marlin = Some(packed);
            if !drop_disabled {
                layer.mlp.up_proj_t = dropped_bf16_stub(device)?;
            }
        }
        MarlinPackKind::DownProj => {
            layer.mlp.down_proj_marlin = Some(packed);
            if !drop_disabled {
                layer.mlp.down_proj_t = dropped_bf16_stub(device)?;
            }
        }
        MarlinPackKind::GdnOutProj => {
            if let GpuAttentionWeights::Linear(ref mut lin) = layer.attention {
                lin.out_proj_marlin = Some(packed);
                if !drop_disabled {
                    lin.out_proj_t = dropped_bf16_stub(device)?;
                }
            }
        }
    }
    Ok(())
}

impl GpuWeights {
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn mtp_weights(&self) -> Result<&MtpGpuWeights> {
        let mtp = self.mtp.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "native MTP requested but checkpoint has no mtp.* tensors \
                 (Qwen3.5-4B includes them)"
            )
        })?;
        mtp.get_or_upload()
    }

    pub fn linear_attention_layers_in_prefix(&self, end_layer: usize) -> usize {
        self.layers
            .iter()
            .take(end_layer.min(self.layers.len()))
            .filter(|layer| matches!(layer.attention, GpuAttentionWeights::Linear(_)))
            .count()
    }

    /// Convert `ModelWeights` (CPU bytes) into candle tensors on the given device.
    ///
    /// `config` is used to precompute the rotary `inv_freq` tensor once so the RoPE
    /// hot path does not re-upload it on every call.
    pub fn from_model_weights(
        weights: &ModelWeights,
        config: &kiln_core::config::ModelConfig,
        device: &Device,
    ) -> Result<Self> {
        // On Metal and on Vulkan-active processes, `embed_tokens` itself
        // is never read past `embedding_lookup_from_weights` (which falls
        // back to `embed_tokens_t` whenever the dims don't match the
        // expected `[vocab, hidden]` shape — the stub case). Materializing
        // both copies costs ~1.3 GB of CPU storage on Qwen3.5-4B BF16
        // for nothing, so collapse to a stub on those backends. On a
        // unified-memory APU this is what keeps Phase 0 from sitting on
        // a duplicate embedding table that the Vulkan side has already
        // mirrored to its own buffer cache.
        let (embed_tokens, embed_tokens_t) = if stub_embed_tokens_after_upload(device) {
            let embed_tokens_t =
                weight_to_transposed_tensor_2d(&weights.embedding.embed_tokens, device)
                    .context("embed_tokens transposed upload")?;
            let embed_tokens = dropped_weight_stub(&weights.embedding.embed_tokens, device)
                .context("embed_tokens stub")?;
            (embed_tokens, embed_tokens_t)
        } else {
            let embed_tokens = weight_to_tensor(&weights.embedding.embed_tokens, device)
                .context("embed_tokens")?;
            let embed_tokens_t =
                cached_transpose_for_weight(&weights.embedding.embed_tokens, &embed_tokens, device)
                    .context("embed_tokens cached transpose")?;
            (embed_tokens, embed_tokens_t)
        };
        let final_norm = weight_to_tensor(&weights.final_norm, device).context("final_norm")?;
        let rotary_inv_freq =
            compute_rotary_inv_freq(config.rotary_dim(), config.rope_theta, device)
                .context("rotary_inv_freq")?;
        let projection_load_cache =
            ProjectionLoadCache::new(device).context("projection load cache")?;
        if projection_load_cache.drops_projection_originals() {
            tracing::info!("projection original tensors are dropped after transposed upload");
        } else if projection_load_cache.drops_projection_transposes() {
            tracing::info!(
                "projection transposed tensors are dropped because Vulkan-native training keeps originals"
            );
        }

        // Per-layer `pack_from_bf16` used to run inline during weight load,
        // serializing ~104 calls (8 × q_proj + 96 × MLP gate/up/down) behind
        // a single thread. At ~58s cold load on the Qwen3.5-4B A6000 build
        // this is a significant fraction of server startup. Sidecar the
        // pack inputs here, batch-pack via rayon after the layer loop, and
        // install results into the per-layer slots.
        let w4a16_enabled = crate::marlin_proj::env_enabled();
        let mut marlin_pack_inputs: Vec<(Tensor, i32)> = Vec::new();
        let mut marlin_pack_meta: Vec<MarlinPackEntry> = Vec::new();

        let mut layers = Vec::with_capacity(weights.layers.len());
        for (i, lw) in weights.layers.iter().enumerate() {
            let ctx = |name: &str| format!("layer {i} {name}");

            let (input_layernorm, post_attention_layernorm, attention) = match &lw.attention {
                crate::weights::AttentionWeights::Full(attn) => {
                    let aux_tensors = aux_tensors_for_load_batch(
                        &[
                            ("input_layernorm", &lw.input_layernorm),
                            ("post_attention_layernorm", &lw.post_attention_layernorm),
                            ("q_norm", &attn.q_norm),
                            ("k_norm", &attn.k_norm),
                        ],
                        device,
                    )
                    .context(ctx("full attention aux tensors"))?;
                    let mut aux_tensors = aux_tensors.into_iter();
                    let input_layernorm =
                        aux_tensors.next().context(ctx("input_layernorm missing"))?;
                    let post_attention_layernorm = aux_tensors
                        .next()
                        .context(ctx("post_attention_layernorm missing"))?;
                    let q_norm = aux_tensors.next().context(ctx("q_norm missing"))?;
                    let k_norm = aux_tensors.next().context(ctx("k_norm missing"))?;

                    let attn_proj = projection_tensors_for_load_batch(
                        &[
                            ("q_proj", &attn.q_proj),
                            ("k_proj", &attn.k_proj),
                            ("v_proj", &attn.v_proj),
                            ("o_proj", &attn.o_proj),
                        ],
                        device,
                        &projection_load_cache,
                    )
                    .context(ctx("attention projection tensors"))?;
                    let mut attn_proj = attn_proj.into_iter();
                    let (q_proj, q_proj_t) = attn_proj.next().context(ctx("q_proj missing"))?;
                    let (k_proj, k_proj_t) = attn_proj.next().context(ctx("k_proj missing"))?;
                    let (v_proj, v_proj_t) = attn_proj.next().context(ctx("v_proj missing"))?;
                    let (o_proj, o_proj_t) = attn_proj.next().context(ctx("o_proj missing"))?;
                    let qkv_proj_t = {
                        #[cfg(feature = "cuda")]
                        {
                            if matches!(device, Device::Cuda(_)) {
                                Some(
                                    Tensor::cat(
                                        &[&q_proj_t, &k_proj_t, &v_proj_t],
                                        candle_core::D::Minus1,
                                    )?
                                    .contiguous()
                                    .context(ctx("qkv_proj_t contiguous"))?,
                                )
                            } else {
                                None
                            }
                        }
                        #[cfg(not(feature = "cuda"))]
                        {
                            None
                        }
                    };
                    // KILN_W4A16=1 opt-in: queue q_proj for the post-loop
                    // Marlin batch pack. The packed weight (and the BF16
                    // drop) are installed after the layer loop via
                    // `install_marlin_packed`, so `q_proj_marlin` starts as
                    // None and `q_proj_t` keeps the BF16 copy until then.
                    if w4a16_enabled {
                        marlin_pack_inputs.push((q_proj_t.clone(), 128));
                        marlin_pack_meta.push(MarlinPackEntry {
                            layer_idx: i,
                            kind: MarlinPackKind::QProj,
                        });
                    }
                    (
                        input_layernorm,
                        post_attention_layernorm,
                        GpuAttentionWeights::Full(GpuFullAttentionWeights {
                            q_proj,
                            k_proj,
                            v_proj,
                            o_proj,
                            q_norm,
                            k_norm,
                            q_proj_t,
                            k_proj_t,
                            v_proj_t,
                            qkv_proj_t,
                            o_proj_t,
                            q_proj_marlin: None,
                        }),
                    )
                }
                crate::weights::AttentionWeights::Linear(attn) => {
                    let aux_tensors = aux_tensors_for_load_batch(
                        &[
                            ("input_layernorm", &lw.input_layernorm),
                            ("post_attention_layernorm", &lw.post_attention_layernorm),
                            ("conv1d", &attn.conv1d),
                            ("gdn_norm", &attn.norm),
                            ("a_log", &attn.a_log),
                            ("dt_bias", &attn.dt_bias),
                        ],
                        device,
                    )
                    .context(ctx("linear attention aux tensors"))?;
                    let mut aux_tensors = aux_tensors.into_iter();
                    let input_layernorm =
                        aux_tensors.next().context(ctx("input_layernorm missing"))?;
                    let post_attention_layernorm = aux_tensors
                        .next()
                        .context(ctx("post_attention_layernorm missing"))?;
                    let conv1d = aux_tensors.next().context(ctx("conv1d missing"))?;
                    let norm = aux_tensors.next().context(ctx("gdn_norm missing"))?;
                    let a_log = aux_tensors.next().context(ctx("a_log missing"))?;
                    let dt_bias = aux_tensors.next().context(ctx("dt_bias missing"))?;
                    let a_log_gates = a_log
                        .to_dtype(DType::BF16)
                        .context(ctx("a_log gates bf16 cache"))?;

                    let attn_proj = projection_tensors_for_load_batch(
                        &[
                            ("in_proj_qkv", &attn.in_proj_qkv),
                            ("in_proj_z", &attn.in_proj_z),
                            ("out_proj", &attn.out_proj),
                            ("in_proj_a", &attn.in_proj_a),
                            ("in_proj_b", &attn.in_proj_b),
                        ],
                        device,
                        &projection_load_cache,
                    )
                    .context(ctx("linear attention projection tensors"))?;
                    let mut attn_proj = attn_proj.into_iter();
                    let (in_proj_qkv, in_proj_qkv_t) =
                        attn_proj.next().context(ctx("in_proj_qkv missing"))?;
                    let (in_proj_z, in_proj_z_t) =
                        attn_proj.next().context(ctx("in_proj_z missing"))?;
                    let (out_proj, out_proj_t) =
                        attn_proj.next().context(ctx("out_proj missing"))?;
                    // KILN_W4A16=1 + KILN_W4A16_GDN_OUT_PROJ=1 opt-in: queue
                    // the GDN out_proj for Marlin batch pack. Gated separately
                    // from the rest because it's the last linear in the GDN
                    // block before the residual add, so int4 here is more
                    // quality-sensitive than the in-projections or the MLP.
                    if w4a16_enabled && crate::marlin_proj::gdn_out_proj_enabled() {
                        marlin_pack_inputs.push((out_proj_t.clone(), 128));
                        marlin_pack_meta.push(MarlinPackEntry {
                            layer_idx: i,
                            kind: MarlinPackKind::GdnOutProj,
                        });
                    }
                    let (in_proj_a, in_proj_a_t) =
                        attn_proj.next().context(ctx("in_proj_a missing"))?;
                    let (in_proj_b, in_proj_b_t) =
                        attn_proj.next().context(ctx("in_proj_b missing"))?;
                    let in_proj_ab_t = {
                        #[cfg(any(feature = "cuda", feature = "metal"))]
                        {
                            let mut should_cache = false;
                            #[cfg(feature = "cuda")]
                            {
                                should_cache |= matches!(device, Device::Cuda(_));
                            }
                            #[cfg(feature = "metal")]
                            {
                                should_cache |= matches!(device, Device::Metal(_));
                            }
                            if should_cache {
                                Some(
                                    Tensor::cat(
                                        &[&in_proj_a_t, &in_proj_b_t],
                                        candle_core::D::Minus1,
                                    )?
                                    .contiguous()
                                    .context(ctx("in_proj_ab_t contiguous"))?,
                                )
                            } else {
                                None
                            }
                        }
                        #[cfg(not(any(feature = "cuda", feature = "metal")))]
                        {
                            None
                        }
                    };
                    (
                        input_layernorm,
                        post_attention_layernorm,
                        GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                            in_proj_qkv,
                            in_proj_z,
                            out_proj,
                            in_proj_a,
                            in_proj_b,
                            conv1d,
                            norm,
                            a_log,
                            a_log_gates,
                            dt_bias,
                            in_proj_qkv_t,
                            in_proj_z_t,
                            in_proj_a_t,
                            in_proj_b_t,
                            in_proj_ab_t,
                            out_proj_t,
                            out_proj_marlin: None,
                        }),
                    )
                }
            };

            let mlp_proj = projection_tensors_for_load_batch(
                &[
                    ("gate_proj", &lw.mlp.gate_proj),
                    ("up_proj", &lw.mlp.up_proj),
                    ("down_proj", &lw.mlp.down_proj),
                ],
                device,
                &projection_load_cache,
            )
            .context(ctx("mlp projection tensors"))?;
            let mut mlp_proj = mlp_proj.into_iter();
            let (gate_proj, gate_proj_t) = mlp_proj.next().context(ctx("gate_proj missing"))?;
            let (up_proj, up_proj_t) = mlp_proj.next().context(ctx("up_proj missing"))?;
            let (down_proj, down_proj_t) = mlp_proj.next().context(ctx("down_proj missing"))?;
            // KILN_W4A16=1 opt-in: queue each MLP projection for the
            // post-loop Marlin batch pack. See the q_proj comment above —
            // the `*_proj_marlin` fields start as None, and
            // `install_marlin_packed` drops `*_proj_t` after the batch runs
            // (unless `KILN_DISABLE_MARLIN_BF16_DROP=1`).
            if w4a16_enabled {
                marlin_pack_inputs.push((gate_proj_t.clone(), 128));
                marlin_pack_meta.push(MarlinPackEntry {
                    layer_idx: i,
                    kind: MarlinPackKind::GateProj,
                });
                marlin_pack_inputs.push((up_proj_t.clone(), 128));
                marlin_pack_meta.push(MarlinPackEntry {
                    layer_idx: i,
                    kind: MarlinPackKind::UpProj,
                });
                marlin_pack_inputs.push((down_proj_t.clone(), 128));
                marlin_pack_meta.push(MarlinPackEntry {
                    layer_idx: i,
                    kind: MarlinPackKind::DownProj,
                });
            }
            let mlp = GpuFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
            };

            layers.push(GpuLayerWeights {
                input_layernorm,
                post_attention_layernorm,
                attention,
                mlp,
            });
        }

        // Batch-pack the queued Marlin projections in parallel. On
        // Qwen3.5-4B this is 8 × q_proj + 96 × MLP = 104 projections. The
        // CPU-bound `quantize_and_pack` work now runs across every
        // available worker thread (rayon's default pool) while the
        // GPU↔CPU copies stay sequential inside
        // `pack_from_bf16_batch`. Set `KILN_DISABLE_PARALLEL_PACK=1` to
        // force the legacy serial pack for A/B measurements or rollback.
        if w4a16_enabled && !marlin_pack_inputs.is_empty() {
            let pack_start = std::time::Instant::now();
            let packed = crate::marlin_proj::pack_from_bf16_batch(&marlin_pack_inputs)
                .context("marlin batch pack")?;
            let pack_elapsed_ms = pack_start.elapsed().as_millis();
            let parallel = !crate::marlin_proj::parallel_pack_disabled();
            let n_inputs = marlin_pack_inputs.len();
            let n_packed = packed.iter().filter(|p| p.is_some()).count();
            tracing::info!(
                n_inputs,
                n_packed,
                pack_elapsed_ms = pack_elapsed_ms as u64,
                parallel,
                "marlin batch pack complete"
            );
            eprintln!(
                "[kiln] marlin batch pack: {n_packed}/{n_inputs} projections in {pack_elapsed_ms} ms ({})",
                if parallel { "parallel" } else { "serial" }
            );

            let drop_disabled = marlin_bf16_drop_disabled();
            for (entry, maybe_packed) in marlin_pack_meta.into_iter().zip(packed.into_iter()) {
                if let Some(p) = maybe_packed {
                    install_marlin_packed(
                        &mut layers[entry.layer_idx],
                        entry.kind,
                        p,
                        device,
                        drop_disabled,
                    )
                    .with_context(|| {
                        format!(
                            "install marlin {:?} on layer {}",
                            entry.kind, entry.layer_idx
                        )
                    })?;
                }
            }
        }

        // Keep MTP routing support visible but do not upload native MTP tensors
        // during model load. The macOS desktop default only uses native MTP for
        // short greedy prompts; long prompts route to skip-layer, so eager MTP
        // upload slows common startup/readiness without warming the hot path.
        let mtp = if let Some(mtp_w) = weights.mtp.as_ref() {
            Some(MtpGpuWeightsSlot::lazy(mtp_w.clone(), device))
        } else {
            weights
                .deferred_mtp
                .as_ref()
                .map(|source| MtpGpuWeightsSlot::lazy_deferred(source.clone(), device))
        };

        if projection_load_cache.drops_projection_originals() && matches!(device, Device::Metal(_))
        {
            device
                .synchronize()
                .context("synchronize after dropping Metal projection originals")?;
            tracing::info!("Metal projection original buffer cache swept after load");
        }

        Ok(Self {
            embed_tokens,
            embed_tokens_t,
            layers,
            final_norm,
            rotary_inv_freq,
            mtp,
        })
    }
}

// ---------------------------------------------------------------------------
// Forward pass primitives
// ---------------------------------------------------------------------------

/// Look up token embeddings from the embedding table.
///
/// `token_ids`: 1-D slice of token IDs.
/// `embed_weights`: [vocab_size, hidden_size] embedding matrix.
///
/// Returns: [seq_len, hidden_size] tensor.
pub fn embedding_lookup(token_ids: &[u32], embed_weights: &Tensor) -> Result<Tensor> {
    let index = Tensor::new(token_ids, embed_weights.device())?;
    let out = embed_weights.index_select(&index, 0)?;
    promote_cpu_activation(out)
}

fn embedding_lookup_with_index(index: &Tensor, embed_weights: &Tensor) -> Result<Tensor> {
    promote_cpu_activation(embed_weights.index_select(index, 0)?)
}

fn embedding_lookup_from_weights(token_ids: &[u32], weights: &GpuWeights) -> Result<Tensor> {
    let t_dims = weights.embed_tokens_t.dims();
    if t_dims.len() == 2 {
        let expected_embed_dims = [t_dims[1], t_dims[0]];
        if weights.embed_tokens.dims() != expected_embed_dims.as_slice() {
            return embedding_lookup_from_transposed(token_ids, &weights.embed_tokens_t);
        }
    }
    embedding_lookup(token_ids, &weights.embed_tokens)
}

fn embedding_lookup_from_weights_with_index(
    index: &Tensor,
    weights: &GpuWeights,
) -> Result<Tensor> {
    let t_dims = weights.embed_tokens_t.dims();
    if t_dims.len() == 2 {
        let expected_embed_dims = [t_dims[1], t_dims[0]];
        if weights.embed_tokens.dims() != expected_embed_dims.as_slice() {
            return embedding_lookup_from_transposed_index(index, &weights.embed_tokens_t);
        }
    }

    embedding_lookup_with_index(index, &weights.embed_tokens)
}

fn embedding_lookup_from_transposed(token_ids: &[u32], embed_tokens_t: &Tensor) -> Result<Tensor> {
    let index = Tensor::new(token_ids, embed_tokens_t.device())?;
    embedding_lookup_from_transposed_index(&index, embed_tokens_t)
}

fn embedding_lookup_from_transposed_index(
    index: &Tensor,
    embed_tokens_t: &Tensor,
) -> Result<Tensor> {
    let gathered = embed_tokens_t.index_select(index, 1)?;
    promote_cpu_activation(gathered.t()?.contiguous()?)
}

/// RMSNorm: x * weight / sqrt(mean(x^2) + eps).
///
/// `x`: [..., hidden_size]
/// `weight`: [hidden_size] (learnable scale)
/// `eps`: small constant for numerical stability (1e-6 for Qwen3.5-4B)
///
/// Returns: same shape as `x`.
///
/// Dispatch (CUDA path; bf16 inputs within the kernel envelope, hidden <= 8192):
///   - Inference tensors that do not participate in autograd use
///     `kiln_rmsnorm_kernel::fused_rmsnorm`, the forward-only single-launch
///     kernel. This path does not build a `CustomOp2` and is therefore not
///     subject to the training saved-tensor VRAM regression described below.
///   - Training/autograd tensors use
///     `kiln_rmsnorm_kernel::fused_rmsnorm_with_autograd` — fused forward
///     kernel + manual CUDA backward kernel routed through `CustomOp2`. The
///     autograd graph saves only `x` and `weight` (not the F32 intermediates
///     that the candle-op chain materializes), shrinking per-layer
///     saved-tensor peak during Phase 10 long-context training.
///   - Auto-gated by total VRAM for autograd only: the `CustomOp2` path runs
///     only on GPUs with ≥ 47 GiB total VRAM (A6000-class and above). On
///     smaller GPUs (A40, RTX 3090/4090, L40, etc.) the dispatch routes through
///     `rms_norm_fallback` automatically. Rationale: PR #638's CustomOp2
///     saved-tensor expansion costs +18.6 GiB peak at T=2048 on A40-class
///     hardware (see `docs/audits/PHASE10_VRAM_REGRESSION_MECHANISM.md`,
///     PR #643). The kernel itself is bit-exact correct; the cost is
///     invisible on A6000 because A6000's allocator pool already sits in
///     the larger profile. Override with `KILN_FORCE_RMSNORM_KERNEL=1` to
///     bypass the gate (useful for benchmarking the regression on smaller
///     GPUs or for forcing the kernel on hardware that detects below the
///     threshold).
///   - `KILN_DISABLE_RMSNORM_BACKWARD=1`: full `rms_norm_fallback` candle-op
///     chain. The forward kernel is not differentiable on its own (it returns
///     a Tensor with no `BackpropOp`), so a "forward-only kernel + autograd
///     backward" hybrid is not a valid path; this kill switch reverts to the
///     pre-Phase-10 baseline (forward AND backward via the candle chain).
///     Intended for isolating the saved-tensor reduction contribution to
///     training peak VRAM. Takes precedence over the auto-VRAM gate.
///   - `KILN_DISABLE_RMSNORM_KERNEL=1`: alias for the above kill switch
///     (also routes through `rms_norm_fallback`). Takes precedence over
///     `KILN_FORCE_RMSNORM_KERNEL` and over the auto-VRAM gate.
///
/// On CPU, non-bf16, out-of-envelope, or non-CUDA backends: falls back to
/// `rms_norm_fallback`. The CPU path keeps the candle-op chain for bit-exact
/// parity with prior trainer tests.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let kernel_disabled = std::env::var("KILN_DISABLE_RMSNORM_KERNEL").is_ok();
        let bwd_disabled = std::env::var("KILN_DISABLE_RMSNORM_BACKWARD").is_ok();
        if !kernel_disabled && !bwd_disabled && kiln_rmsnorm_kernel::supports(x, weight) {
            if !x.track_op() && !weight.track_op() {
                return kiln_rmsnorm_kernel::fused_rmsnorm(x, weight, eps as f32)
                    .context("fused_rmsnorm inference forward failed");
            }
            if should_use_fused_rmsnorm() {
                return kiln_rmsnorm_kernel::fused_rmsnorm_with_autograd(x, weight, eps as f32)
                    .context("fused_rmsnorm_with_autograd CustomOp2 failed");
            }
        }
    }
    #[cfg(feature = "metal")]
    {
        if !x.track_op()
            && !weight.track_op()
            && crate::backend::metal::metal_rms_norm_supports(x, weight)
        {
            return crate::backend::metal::metal_rms_norm_bf16(x, weight, eps as f32)
                .context("metal rms_norm kernel failed");
        }
    }
    // Vulkan inference path: leaf-fast forward kernel. Skipped for
    // autograd-tracked tensors because the leaf would drop the
    // gradient — the autograd-safe CustomOp1 wrapper below handles
    // those instead (when its separate opt-in is set).
    #[cfg(feature = "vulkan")]
    if vulkan_rmsnorm_forward_inference_enabled()
        && crate::backend::vulkan_active()
        && matches!(x.device(), Device::Cpu)
        && matches!(weight.device(), Device::Cpu)
        && weight.is_contiguous()
        && !weight.track_op()
    {
        if !x.track_op() {
            if let Some(out) = try_vulkan_rmsnorm_forward(x, weight, eps as f32)? {
                return Ok(out);
            }
        } else if vulkan_rmsnorm_training_enabled_for(x) {
            if let Some(out) = try_vulkan_rmsnorm_autograd(x, weight, eps as f32)? {
                return Ok(out);
            }
        }
    }
    rms_norm_fallback(x, weight, eps)
}

/// Tristate env-var resolution for the autograd-safe RMSNorm path.
///
/// `KILN_VULKAN_RMSNORM_TRAINING=1` forces on, `=0` forces off,
/// otherwise the auto-heuristic decides based on the per-row count
/// of `x` — the same constant-overhead-vs-compute trade-off that the
/// FLCE auto-heuristic resolves: at small T the per-call dispatch
/// overhead (upload x + readback dx) exceeds the kernel's compute
/// savings vs the candle CPU `broadcast_mul` chain. Hardware
/// measurement on Strix Halo at T=244 (~30 active rows × ~64 RMSNorm
/// calls per forward+backward) put the autograd RMSNorm at +8 s wall
/// vs the candle fallback. Crossover where it becomes a net win is
/// expected around T=1500-2500.
///
/// Threshold: enable when the row count of x (= batch × seq_len) is
/// at least `RMSNORM_AUTO_ROW_THRESHOLD = 1024`. Tunable via this
/// constant; documented inline so the next data-driven measurement
/// can move it.
#[cfg(feature = "vulkan")]
fn vulkan_rmsnorm_training_enabled_for(x: &Tensor) -> bool {
    if let Some(forced) = kiln_core::env_flag::env_tristate("KILN_VULKAN_RMSNORM_TRAINING") {
        return forced;
    }
    const RMSNORM_AUTO_ROW_THRESHOLD: usize = 1024;
    let dims = x.shape().dims();
    if dims.is_empty() {
        return false;
    }
    let row_count: usize = dims[..dims.len() - 1].iter().product();
    row_count >= RMSNORM_AUTO_ROW_THRESHOLD
}

/// `KILN_VULKAN_RMSNORM=0` opts the inference RMSNorm Vulkan path off.
/// Default: enabled.
#[cfg(feature = "vulkan")]
fn vulkan_rmsnorm_forward_inference_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_VULKAN_RMSNORM")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(true)
    })
}

/// Inference-only Vulkan RMSNorm dispatch. Promotes inputs to F32,
/// dispatches the kernel, casts result back to the input dtype.
/// Returns `Ok(None)` when preconditions don't fit (caller falls back).
#[cfg(feature = "vulkan")]
fn try_vulkan_rmsnorm_forward(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Option<Tensor>> {
    let Some(vk_device) = vulkan_device_handle() else {
        return Ok(None);
    };
    let in_dtype = x.dtype();
    let x_f32 = if in_dtype == DType::F32 {
        x.contiguous()?
    } else {
        x.to_dtype(DType::F32)?.contiguous()?
    };
    let w_f32 = if weight.dtype() == DType::F32 {
        weight.clone()
    } else {
        weight.to_dtype(DType::F32)?
    };
    let out_f32 = kiln_vulkan_kernel::kernels::dispatch_qwen_rmsnorm_forward(
        vk_device.as_ref(),
        &x_f32,
        &w_f32,
        eps,
    )?;
    let out = if out_f32.dtype() == in_dtype {
        out_f32
    } else {
        out_f32.to_dtype(in_dtype)?
    };
    Ok(Some(out))
}

/// Process-cached Vulkan device handle. The first call constructs +
/// caches the handle; subsequent calls just clone the Arc.
#[cfg(feature = "vulkan")]
fn vulkan_device_handle() -> Option<std::sync::Arc<kiln_vulkan_kernel::VulkanDevice>> {
    static VK_DEVICE: std::sync::OnceLock<
        Option<std::sync::Arc<kiln_vulkan_kernel::VulkanDevice>>,
    > = std::sync::OnceLock::new();
    VK_DEVICE
        .get_or_init(|| {
            kiln_vulkan_kernel::VulkanDevice::new()
                .ok()
                .map(std::sync::Arc::new)
        })
        .clone()
}

/// Autograd-safe Vulkan RMSNorm: wraps `dispatch_qwen_rmsnorm_forward` +
/// `dispatch_qwen_rmsnorm_backward` in a `CustomOp1` so `loss.backward()`
/// flows the gradient through `dL/dx` correctly.
///
/// The `weight` is captured into op state because Qwen3.5 base RMSNorm
/// weights are frozen during LoRA training — only `x` participates in
/// autograd.
#[cfg(feature = "vulkan")]
fn try_vulkan_rmsnorm_autograd(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Option<Tensor>> {
    use candle_core::{CpuStorage, CustomOp1, Layout, Shape, Storage};

    let Some(vk_device) = vulkan_device_handle() else {
        return Ok(None);
    };
    let in_dtype = x.dtype();

    struct VulkanRmsNormOp {
        vk_device: std::sync::Arc<kiln_vulkan_kernel::VulkanDevice>,
        weight: Tensor, // captured frozen weight
        eps: f32,
        out_dtype: DType,
    }
    impl std::fmt::Debug for VulkanRmsNormOp {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("VulkanRmsNormOp")
                .field("eps", &self.eps)
                .field("out_dtype", &self.out_dtype)
                .field("hidden", &self.weight.dims())
                .finish()
        }
    }
    impl CustomOp1 for VulkanRmsNormOp {
        fn name(&self) -> &'static str {
            "kiln-vulkan-qwen-rmsnorm"
        }
        fn cpu_fwd(
            &self,
            s_x: &CpuStorage,
            l_x: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            let storage = Storage::Cpu(s_x.clone());
            let x_tensor = Tensor::from_storage(
                storage,
                Shape::from(l_x.shape().dims()),
                candle_core::op::BackpropOp::none(),
                false,
            );
            let x_f32 = if x_tensor.dtype() == DType::F32 {
                x_tensor.contiguous()?
            } else {
                x_tensor.to_dtype(DType::F32)?.contiguous()?
            };
            let w_f32 = if self.weight.dtype() == DType::F32 {
                self.weight.clone()
            } else {
                self.weight.to_dtype(DType::F32).map_err(|e| {
                    candle_core::Error::Msg(format!("rmsnorm fwd weight→f32: {e:?}"))
                })?
            };
            let out_f32 = kiln_vulkan_kernel::kernels::dispatch_qwen_rmsnorm_forward(
                self.vk_device.as_ref(),
                &x_f32,
                &w_f32,
                self.eps,
            )
            .map_err(|e| candle_core::Error::Msg(format!("rmsnorm fwd dispatch: {e:?}")))?;
            let out = if out_f32.dtype() == self.out_dtype {
                out_f32
            } else {
                out_f32
                    .to_dtype(self.out_dtype)
                    .map_err(|e| candle_core::Error::Msg(format!("rmsnorm fwd cast: {e:?}")))?
            };
            let storage = out
                .storage_and_layout()
                .0
                .try_clone(out.layout())
                .map_err(|e| {
                    candle_core::Error::Msg(format!("rmsnorm fwd storage clone: {e:?}"))
                })?;
            let cpu_storage = match storage {
                Storage::Cpu(s) => s,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "rmsnorm fwd: expected CPU storage from kernel result".into(),
                    ));
                }
            };
            Ok((cpu_storage, Shape::from(out.dims())))
        }
        fn bwd(
            &self,
            x: &Tensor,
            _y: &Tensor,
            grad_y: &Tensor,
        ) -> candle_core::Result<Option<Tensor>> {
            let x_f32 = if x.dtype() == DType::F32 {
                x.clone()
            } else {
                x.to_dtype(DType::F32)?
            };
            let w_f32 = if self.weight.dtype() == DType::F32 {
                self.weight.clone()
            } else {
                self.weight.to_dtype(DType::F32)?
            };
            let grad_y_f32 = if grad_y.dtype() == DType::F32 {
                grad_y.clone()
            } else {
                grad_y.to_dtype(DType::F32)?
            };
            let dx_f32 = kiln_vulkan_kernel::kernels::dispatch_qwen_rmsnorm_backward(
                self.vk_device.as_ref(),
                &x_f32,
                &w_f32,
                &grad_y_f32,
                self.eps,
            )
            .map_err(|e| candle_core::Error::Msg(format!("rmsnorm bwd dispatch: {e:?}")))?;
            let dx = if self.out_dtype == DType::F32 {
                dx_f32
            } else {
                dx_f32
                    .to_dtype(self.out_dtype)
                    .map_err(|e| candle_core::Error::Msg(format!("rmsnorm bwd cast: {e:?}")))?
            };
            Ok(Some(dx))
        }
    }

    let op = VulkanRmsNormOp {
        vk_device,
        weight: weight.clone(),
        eps,
        out_dtype: in_dtype,
    };
    let x_contig = x.contiguous()?;
    let out = x_contig.apply_op1(op)?;
    Ok(Some(out))
}

/// Candle-op reference RMSNorm. Kept as the CPU path and as the correctness
/// oracle for the fused CUDA kernel. Matches HF semantics exactly:
/// `out = (1 + w) * x * rsqrt(mean(x^2) + eps)` with F32 reduction and epilogue.
pub fn rms_norm_fallback(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rms_inv = (variance + eps)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&rms_inv)?;
    // Qwen3.5 RMSNorm stores weights centered around 0 and applies as (1 + w) * x_normed.
    // Keep everything in F32 for precision (matches HF: `output * (1.0 + self.weight.float())`),
    // then cast back to input dtype at the end.
    let w_f32 = weight.to_dtype(DType::F32)?;
    let w_plus_one = (w_f32.ones_like()? + w_f32)?;
    let out = normed.broadcast_mul(&w_plus_one)?;
    Ok(out.to_dtype(x.dtype())?)
}

/// Phase C42: capture the minimal layer-1 input-layernorm intermediates needed
/// to distinguish "bad residual input arrives at block 1" from "the residual
/// input is clean but the RMSNorm math diverges". This intentionally mirrors
/// the fallback RMSNorm formula instead of widening the general tracing API.
fn capture_c42_layer1_input_norm_taps(x: &Tensor, weight: &Tensor, eps: f64) -> Result<()> {
    crate::mtp_debug::capture_c42_layer1_norm_tap("layer_1_residual_input", x)?;
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rms_inv = (variance + eps)?.sqrt()?.recip()?;
    crate::mtp_debug::capture_c42_layer1_norm_tap("layer_1_input_norm_rms_inv", &rms_inv)?;
    let pre_weight = x_f32.broadcast_mul(&rms_inv)?;
    crate::mtp_debug::capture_c42_layer1_norm_tap("layer_1_input_norm_pre_weight", &pre_weight)?;
    let w_f32 = weight.to_dtype(DType::F32)?;
    let w_plus_one = (w_f32.ones_like()? + w_f32)?;
    let post = pre_weight.broadcast_mul(&w_plus_one)?.to_dtype(x.dtype())?;
    crate::mtp_debug::capture_c42_layer1_norm_tap("layer_1_post_input_norm", &post)?;
    Ok(())
}

/// Phase C43: keep the C42 layer-1 norm boundary context, but split the
/// pre-weight multiply into the existing broadcast path and an independently
/// computed scalar-affine equivalent so the replay dump can distinguish
/// "broadcast/layout/row-selection bug" from "the normalized values
/// themselves are already wrong".
fn capture_c43_layer1_preweight_taps(x: &Tensor, weight: &Tensor, eps: f64) -> Result<()> {
    crate::mtp_debug::capture_c43_layer1_preweight_tap("layer_1_residual_input", x)?;
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rms_inv = (variance + eps)?.sqrt()?.recip()?;
    crate::mtp_debug::capture_c43_layer1_preweight_tap("layer_1_input_norm_rms_inv", &rms_inv)?;

    let pre_weight_broadcast = x_f32.broadcast_mul(&rms_inv)?;
    crate::mtp_debug::capture_c43_layer1_preweight_tap(
        "layer_1_input_norm_pre_weight_broadcast_mul",
        &pre_weight_broadcast,
    )?;

    let (batch, seq_len, hidden) = x_f32
        .dims3()
        .context("C43 pre-weight audit expects layer-1 hidden to be [batch, seq, hidden]")?;
    let (r_batch, r_seq, r_hidden) = rms_inv
        .dims3()
        .context("C43 pre-weight audit expects rms_inv to be [batch, seq, 1]")?;
    anyhow::ensure!(
        (batch, seq_len, r_hidden) == (r_batch, r_seq, 1),
        "C43 pre-weight audit shape mismatch: x={batch}x{seq_len}x{hidden}, rms_inv={r_batch}x{r_seq}x{r_hidden}"
    );
    let mut batch_slices = Vec::with_capacity(batch);
    for batch_idx in 0..batch {
        let x_batch = x_f32.narrow(0, batch_idx, 1)?;
        let rms_batch = rms_inv.narrow(0, batch_idx, 1)?;
        let mut seq_slices = Vec::with_capacity(seq_len);
        for seq_idx in 0..seq_len {
            let x_row = x_batch.narrow(1, seq_idx, 1)?;
            let scale = rms_batch.narrow(1, seq_idx, 1)?;
            let scale_vals = scale.flatten_all()?.to_vec1::<f32>()?;
            anyhow::ensure!(
                scale_vals.len() == 1,
                "C43 pre-weight audit expected one rms_inv scalar per row, got {}",
                scale_vals.len()
            );
            seq_slices.push(x_row.affine(scale_vals[0] as f64, 0.0)?);
        }
        let seq_refs: Vec<&Tensor> = seq_slices.iter().collect();
        batch_slices.push(Tensor::cat(&seq_refs, 1)?);
    }
    let batch_refs: Vec<&Tensor> = batch_slices.iter().collect();
    let pre_weight_scalar_affine = Tensor::cat(&batch_refs, 0)?;
    crate::mtp_debug::capture_c43_layer1_preweight_tap(
        "layer_1_input_norm_pre_weight_scalar_affine",
        &pre_weight_scalar_affine,
    )?;

    let w_f32 = weight.to_dtype(DType::F32)?;
    let w_plus_one = (w_f32.ones_like()? + w_f32)?;
    let post = pre_weight_broadcast
        .broadcast_mul(&w_plus_one)?
        .to_dtype(x.dtype())?;
    crate::mtp_debug::capture_c43_layer1_preweight_tap("layer_1_post_input_norm", &post)?;
    Ok(())
}

/// Phase C44: capture only the last replay row after `x.to_dtype(F32)`, the
/// matching `rms_inv` scalar for that row, and the normalized row after
/// applying the shared-good scalar. This distinguishes "bad row before
/// scaling" from "good row, bad normalization application" without re-dumping
/// the full C43 tensors.
fn capture_c44_layer1_f32_row_taps(x: &Tensor, eps: f64) -> Result<()> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let (batch, seq_len, _hidden) = x_f32
        .dims3()
        .context("C44 row audit expects layer-1 hidden to be [batch, seq, hidden]")?;
    anyhow::ensure!(seq_len > 0, "C44 row audit requires non-empty sequence");

    let last_row = x_f32.narrow(1, seq_len - 1, 1)?.contiguous()?;
    crate::mtp_debug::capture_c44_layer1_f32_row_tap("layer_1_residual_input_f32_row", &last_row)?;

    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rms_inv = (variance + eps)?.sqrt()?.recip()?;
    let rms_inv_row = rms_inv.narrow(1, seq_len - 1, 1)?.contiguous()?;
    crate::mtp_debug::capture_c44_layer1_f32_row_tap(
        "layer_1_input_norm_rms_inv_scalar",
        &rms_inv_row,
    )?;

    let mut batch_rows = Vec::with_capacity(batch);
    for batch_idx in 0..batch {
        let x_row = last_row.narrow(0, batch_idx, 1)?;
        let scale = rms_inv_row.narrow(0, batch_idx, 1)?;
        let scale_vals = scale.flatten_all()?.to_vec1::<f32>()?;
        anyhow::ensure!(
            scale_vals.len() == 1,
            "C44 row audit expected one rms_inv scalar per batch row, got {}",
            scale_vals.len()
        );
        batch_rows.push(x_row.affine(scale_vals[0] as f64, 0.0)?);
    }
    let batch_refs: Vec<&Tensor> = batch_rows.iter().collect();
    let normalized_row = Tensor::cat(&batch_refs, 0)?;
    crate::mtp_debug::capture_c44_layer1_f32_row_tap(
        "layer_1_input_norm_pre_weight_row_scalar_affine",
        &normalized_row,
    )?;
    Ok(())
}

fn c45_layer1_row_replay_tensors(
    x: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let (batch, seq_len, hidden) = x_f32
        .dims3()
        .context("C45 row audit expects layer-1 hidden to be [batch, seq, hidden]")?;
    anyhow::ensure!(seq_len > 0, "C45 row audit requires non-empty sequence");

    let last_row = x_f32.narrow(1, seq_len - 1, 1)?.contiguous()?;
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rms_inv = (variance + eps)?.sqrt()?.recip()?;
    let rms_inv_row = rms_inv.narrow(1, seq_len - 1, 1)?.contiguous()?;

    let mut extracted_scalars = Vec::with_capacity(batch);
    for batch_idx in 0..batch {
        let scale = rms_inv_row.narrow(0, batch_idx, 1)?;
        let scale_vals = scale.flatten_all()?.to_vec1::<f32>()?;
        anyhow::ensure!(
            scale_vals.len() == 1,
            "C45 row audit expected one rms_inv scalar per batch row, got {}",
            scale_vals.len()
        );
        extracted_scalars.push(scale_vals[0]);
    }

    let extracted_scalar_values = extracted_scalars;
    let extracted_scalars =
        Tensor::from_slice(&extracted_scalar_values, (batch,), &Device::Cpu)?.contiguous()?;
    let last_row_values = last_row.reshape((batch, hidden))?.contiguous()?;
    let broadcast_output = last_row.broadcast_mul(&rms_inv_row)?.contiguous()?;

    let scalar_values = broadcast_output.reshape((batch, hidden))?.contiguous()?;
    let reconstructed = scalar_values.reshape((batch, 1, hidden))?.contiguous()?;
    Ok((
        rms_inv_row,
        extracted_scalars,
        last_row_values,
        broadcast_output,
        scalar_values,
        reconstructed,
    ))
}

/// Phase C45: keep the audit strictly inside the previously-bad row-local
/// scalar multiply so the replay dump can distinguish "the row-local scalar
/// tensor is fine", "the scalar extraction path already drifts", "the actual
/// multiply introduces the drift", or "the flattened production multiply
/// output stays shared-good and divergence only appears when reconstructing the
/// row-shaped output".
fn capture_c45_layer1_row_taps(x: &Tensor, eps: f64) -> Result<()> {
    let (
        rms_inv_row,
        extracted_scalars,
        last_row_values,
        broadcast_output,
        scalar_values,
        reconstructed,
    ) = c45_layer1_row_replay_tensors(x, eps)?;
    crate::mtp_debug::capture_c45_layer1_row_tap(
        "layer_1_input_norm_rms_inv_scalar",
        &rms_inv_row,
    )?;
    crate::mtp_debug::capture_c45_layer1_row_tap(
        "layer_1_input_norm_rms_inv_scalar_extracted_values",
        &extracted_scalars,
    )?;
    crate::mtp_debug::capture_c45_layer1_row_tap(
        "layer_1_input_norm_last_row_flat_values",
        &last_row_values,
    )?;
    crate::mtp_debug::capture_c45_layer1_row_tap(
        "layer_1_input_norm_pre_weight_row_broadcast_output",
        &broadcast_output,
    )?;
    crate::mtp_debug::capture_c45_layer1_row_tap(
        "layer_1_input_norm_pre_weight_row_scalar_values",
        &scalar_values,
    )?;
    crate::mtp_debug::capture_c45_layer1_row_tap(
        "layer_1_input_norm_pre_weight_row_reconstructed",
        &reconstructed,
    )?;
    Ok(())
}

fn c46_layer1_row_provenance_tensors(
    x: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let (_batch, seq_len, hidden) = x
        .dims3()
        .context("C46 row provenance expects layer-1 hidden to be [batch, seq, hidden]")?;
    anyhow::ensure!(
        seq_len > 0,
        "C46 row provenance requires non-empty sequence"
    );

    let selected_row = x.narrow(1, seq_len - 1, 1)?;
    let selected_row_f32 = selected_row.to_dtype(DType::F32)?;
    let selected_row_contiguous = selected_row_f32.contiguous()?;
    let selected_row_flat = selected_row_contiguous
        .reshape(((), hidden))?
        .contiguous()?;

    let x_f32 = x.to_dtype(DType::F32)?;
    let (_batch, seq_len, hidden) = x_f32
        .dims3()
        .context("C46 row provenance expects f32 layer-1 hidden to be [batch, seq, hidden]")?;
    let c45_last_row = x_f32
        .narrow(1, seq_len - 1, 1)?
        .contiguous()?
        .reshape(((), hidden))?
        .contiguous()?;

    Ok((
        selected_row,
        selected_row_f32,
        selected_row_contiguous,
        selected_row_flat,
        c45_last_row,
    ))
}

/// Phase C46: bisect the row-side operand provenance feeding C45's
/// `layer_1_input_norm_last_row_flat_values` by splitting row selection,
/// dtype promotion, contiguous materialization, flattening, and the exact C45
/// operand reconstruction into separate taps.
fn capture_c46_layer1_row_provenance_taps(x: &Tensor) -> Result<()> {
    let (selected_row, selected_row_f32, selected_row_contiguous, selected_row_flat, c45_last_row) =
        c46_layer1_row_provenance_tensors(x)?;
    crate::mtp_debug::capture_c46_layer1_row_provenance_tap(
        "layer_1_input_norm_selected_row_before_rmsnorm",
        &selected_row,
    )?;
    crate::mtp_debug::capture_c46_layer1_row_provenance_tap(
        "layer_1_input_norm_selected_row_after_f32_cast",
        &selected_row_f32,
    )?;
    crate::mtp_debug::capture_c46_layer1_row_provenance_tap(
        "layer_1_input_norm_selected_row_after_contiguous",
        &selected_row_contiguous,
    )?;
    crate::mtp_debug::capture_c46_layer1_row_provenance_tap(
        "layer_1_input_norm_selected_row_after_flatten",
        &selected_row_flat,
    )?;
    crate::mtp_debug::capture_c46_layer1_row_provenance_tap(
        "layer_1_input_norm_last_row_flat_values",
        &c45_last_row,
    )?;
    Ok(())
}

/// Apply Rotary Position Embeddings (RoPE) to query and key tensors.
///
/// `q`: [batch, seq_len, num_heads, head_dim]
/// `k`: [batch, seq_len, num_kv_heads, head_dim]
/// `positions`: position index for each token in the sequence (length = seq_len)
/// `head_dim`: dimension of each attention head
/// `rotary_dim`: number of head dimensions to apply rotation to (the rest pass through unchanged).
///   For Qwen3.5-4B: 64 (partial_rotary_factor=0.25, so 0.25 * 256 = 64).
/// `inv_freq`: cached frequency table of shape `[rotary_dim / 2]` (F32 on same device as `q`/`k`).
///   Build once via [`compute_rotary_inv_freq`] and reuse across calls.
///
/// Returns: (rotated_q, rotated_k) with same shapes.
pub fn rotary_embedding(
    q: &Tensor,
    k: &Tensor,
    positions: &[u32],
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let device = q.device();

    // Position tensor
    let pos_f32: Vec<f32> = positions.iter().map(|&p| p as f32).collect();
    let pos = Tensor::new(pos_f32.as_slice(), device)?.unsqueeze(1)?; // [seq_len, 1]

    // Outer product: [seq_len, half_rotary]
    let freqs = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    let cos = freqs.cos()?; // [seq_len, half_rotary]
    let sin = freqs.sin()?; // [seq_len, half_rotary]

    let rotated_q = apply_rope(q, &cos, &sin, head_dim, rotary_dim)?;
    let rotated_k = apply_rope(k, &cos, &sin, head_dim, rotary_dim)?;

    Ok((rotated_q, rotated_k))
}

/// Same as [`rotary_embedding`] but accepts positions as a pre-allocated GPU tensor
/// instead of a CPU slice. This is critical for CUDA graph compatibility: the tensor's
/// GPU address stays stable across graph replays, and its contents can be updated via
/// `cudaMemcpyAsync` outside the captured graph.
///
/// `positions_tensor`: f32 tensor on device, shape [seq_len]
/// `inv_freq`: cached frequency table, shape `[rotary_dim / 2]`, F32 on device.
pub fn rotary_embedding_from_tensor(
    q: &Tensor,
    k: &Tensor,
    positions_tensor: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // positions_tensor is [seq_len], unsqueeze to [seq_len, 1]
    let pos = positions_tensor.unsqueeze(1)?;

    let freqs = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    #[cfg(feature = "cuda")]
    {
        if !cuda_fused_rotary_qk_disabled()
            && kiln_rmsnorm_kernel::supports_rotary_qk(q, k, &cos, &sin, head_dim, rotary_dim)
        {
            return kiln_rmsnorm_kernel::fused_rotary_qk(q, k, &cos, &sin, head_dim, rotary_dim)
                .context("cuda fused rotary qk kernel failed");
        }
    }

    let rotated_q = apply_rope(q, &cos, &sin, head_dim, rotary_dim)?;
    let rotated_k = apply_rope(k, &cos, &sin, head_dim, rotary_dim)?;

    Ok((rotated_q, rotated_k))
}

fn rotary_tables_from_tensor(
    positions_tensor: &Tensor,
    inv_freq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let pos = positions_tensor.unsqueeze(1)?;
    let freqs = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

fn rotary_embedding_from_tables(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        if !cuda_fused_rotary_qk_disabled()
            && kiln_rmsnorm_kernel::supports_rotary_qk(q, k, cos, sin, head_dim, rotary_dim)
        {
            return kiln_rmsnorm_kernel::fused_rotary_qk(q, k, cos, sin, head_dim, rotary_dim)
                .context("cuda fused rotary qk kernel failed");
        }
    }

    #[cfg(feature = "metal")]
    {
        if crate::backend::metal::metal_rotary_embedding_supports(
            q, k, cos, sin, head_dim, rotary_dim,
        ) {
            return crate::backend::metal::metal_rotary_embedding_bf16(
                q, k, cos, sin, head_dim, rotary_dim,
            )
            .context("metal rotary embedding kernel failed");
        }
    }
    let rotated_q = apply_rope(q, cos, sin, head_dim, rotary_dim)?;
    let rotated_k = apply_rope(k, cos, sin, head_dim, rotary_dim)?;
    Ok((rotated_q, rotated_k))
}

/// Apply the rotation to a single tensor, supporting partial rotary embeddings.
/// `x`: [batch, seq_len, num_heads, head_dim]
/// `cos`, `sin`: [seq_len, half_rotary]
/// `head_dim`: total dimension per head
/// `rotary_dim`: number of dimensions to rotate (must be even). The first `rotary_dim` dims
///   are rotated; the remaining `head_dim - rotary_dim` dims pass through unchanged.
fn apply_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if !cuda_fused_rotary_qk_disabled()
            && cuda_rotary_one_training_bf16_supported(x, cos, sin, head_dim, rotary_dim)
        {
            return x
                .apply_op3(
                    cos,
                    sin,
                    CudaRotaryOneBf16 {
                        head_dim,
                        rotary_dim,
                    },
                )
                .context("cuda rotary one CustomOp3");
        }
    }

    let half_rotary = rotary_dim / 2;
    let x_dtype = x.dtype();

    // Work in f32 for precision
    let x = x.to_dtype(DType::F32)?;

    // Split into rotary portion and passthrough portion
    let x_rot = x.narrow(candle_core::D::Minus1, 0, rotary_dim)?; // [..., :rotary_dim]
    let x_pass = if rotary_dim < head_dim {
        Some(x.narrow(candle_core::D::Minus1, rotary_dim, head_dim - rotary_dim)?) // [..., rotary_dim:]
    } else {
        None
    };

    // Split rotary portion into two halves
    let x1 = x_rot.narrow(candle_core::D::Minus1, 0, half_rotary)?; // [..., :half_rotary]
    let x2 = x_rot.narrow(candle_core::D::Minus1, half_rotary, half_rotary)?; // [..., half_rotary:rotary_dim]

    // cos/sin are [seq_len, half_rotary], need to broadcast to [batch, seq_len, num_heads, half_rotary]
    // Reshape to [1, seq_len, 1, half_rotary]
    let cos = cos.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;

    // Standard RoPE rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

    // Concatenate rotated dims + passthrough dims
    let out = match x_pass {
        Some(pass) => Tensor::cat(&[&r1, &r2, &pass], candle_core::D::Minus1)?,
        None => Tensor::cat(&[&r1, &r2], candle_core::D::Minus1)?,
    };
    Ok(out.to_dtype(x_dtype)?)
}

#[cfg(feature = "cuda")]
fn cuda_rotary_one_training_bf16_supported(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> bool {
    if !matches!(x.device(), Device::Cuda(_))
        || !matches!(cos.device(), Device::Cuda(_))
        || !matches!(sin.device(), Device::Cuda(_))
        || x.dtype() != DType::BF16
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
        || !x.is_contiguous()
        || !cos.is_contiguous()
        || !sin.is_contiguous()
        || x.rank() != 4
        || rotary_dim == 0
        || rotary_dim > head_dim
        || rotary_dim % 2 != 0
    {
        return false;
    }
    let dims = x.dims();
    let seq_len = dims[1];
    dims[3] == head_dim
        && cos.dims() == [seq_len, rotary_dim / 2]
        && sin.dims() == [seq_len, rotary_dim / 2]
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
struct CudaRotaryOneBf16 {
    head_dim: usize,
    rotary_dim: usize,
}

#[cfg(feature = "cuda")]
impl CustomOp3 for CudaRotaryOneBf16 {
    fn name(&self) -> &'static str {
        "kiln-cuda-rotary-one-bf16"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
        s_cos: &CpuStorage,
        l_cos: &Layout,
        s_sin: &CpuStorage,
        l_sin: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        if !l_x.is_contiguous()
            || !l_cos.is_contiguous()
            || !l_sin.is_contiguous()
            || l_x.start_offset() != 0
            || l_cos.start_offset() != 0
            || l_sin.start_offset() != 0
        {
            candle_core::bail!("CudaRotaryOneBf16 CPU fallback requires compact contiguous inputs");
        }
        let x = Tensor::from_storage(
            Storage::Cpu(s_x.clone()),
            Shape::from(l_x.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let cos = Tensor::from_storage(
            Storage::Cpu(s_cos.clone()),
            Shape::from(l_cos.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let sin = Tensor::from_storage(
            Storage::Cpu(s_sin.clone()),
            Shape::from(l_sin.dims().to_vec()),
            BackpropOp::none(),
            false,
        );
        let out = apply_rope(&x, &cos, &sin, self.head_dim, self.rotary_dim).map_err(|e| {
            candle_core::Error::Msg(format!("CudaRotaryOneBf16 CPU fallback: {e:?}"))
        })?;
        let (storage, layout) = out.storage_and_layout();
        let storage = storage.try_clone(layout)?;
        match storage {
            Storage::Cpu(storage) => Ok((storage, Shape::from(l_x.dims().to_vec()))),
            _ => candle_core::bail!("CudaRotaryOneBf16 CPU fallback produced non-CPU storage"),
        }
    }

    fn cuda_fwd(
        &self,
        s_x: &CudaStorage,
        l_x: &Layout,
        s_cos: &CudaStorage,
        l_cos: &Layout,
        s_sin: &CudaStorage,
        l_sin: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        if !l_x.is_contiguous() || !l_cos.is_contiguous() || !l_sin.is_contiguous() {
            candle_core::bail!("CudaRotaryOneBf16 CUDA path requires contiguous inputs");
        }
        let out_storage = s_x.try_clone(l_x)?;
        let out_shape = Shape::from(l_x.dims().to_vec());
        let out_layout = Layout::contiguous(out_shape.clone());
        kiln_rmsnorm_kernel::rotary_one_bf16_storage(
            &out_storage,
            &out_layout,
            s_x,
            l_x,
            s_cos,
            l_cos,
            s_sin,
            l_sin,
            self.head_dim,
            self.rotary_dim,
        )
        .map_err(|e| candle_core::Error::Msg(format!("CudaRotaryOneBf16 CUDA fwd: {e:?}")))?;
        Ok((out_storage, out_shape))
    }

    fn bwd(
        &self,
        _x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        _res: &Tensor,
        grad_y: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        if kiln_rmsnorm_kernel::supports_rotary_one_bwd_bf16(
            grad_y,
            cos,
            sin,
            self.head_dim,
            self.rotary_dim,
        ) {
            let grad_x = kiln_rmsnorm_kernel::rotary_one_bwd_bf16(
                grad_y,
                cos,
                sin,
                self.head_dim,
                self.rotary_dim,
            )
            .map_err(|e| candle_core::Error::Msg(format!("CudaRotaryOneBf16 CUDA bwd: {e:?}")))?;
            return Ok((Some(grad_x), None, None));
        }
        let grad_x = rotary_one_backward(grad_y, cos, sin, self.head_dim, self.rotary_dim)
            .map_err(|e| candle_core::Error::Msg(format!("CudaRotaryOneBf16 bwd: {e:?}")))?;
        Ok((Some(grad_x), None, None))
    }
}

#[cfg(feature = "cuda")]
fn rotary_one_backward(
    grad_y: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Tensor> {
    let half_rotary = rotary_dim / 2;
    let grad_dtype = grad_y.dtype();
    let grad = grad_y.to_dtype(DType::F32)?;
    let grad_rot = grad.narrow(candle_core::D::Minus1, 0, rotary_dim)?;
    let grad_pass = if rotary_dim < head_dim {
        Some(grad.narrow(candle_core::D::Minus1, rotary_dim, head_dim - rotary_dim)?)
    } else {
        None
    };

    let g1 = grad_rot.narrow(candle_core::D::Minus1, 0, half_rotary)?;
    let g2 = grad_rot.narrow(candle_core::D::Minus1, half_rotary, half_rotary)?;
    let cos = cos.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;

    let dx1 = (g1.broadcast_mul(&cos)? + g2.broadcast_mul(&sin)?)?;
    let dx2 = (g2.broadcast_mul(&cos)? - g1.broadcast_mul(&sin)?)?;
    let out = match grad_pass {
        Some(pass) => Tensor::cat(&[&dx1, &dx2, &pass], candle_core::D::Minus1)?,
        None => Tensor::cat(&[&dx1, &dx2], candle_core::D::Minus1)?,
    };
    Ok(out.to_dtype(grad_dtype)?)
}

/// SwiGLU feed-forward network.
///
/// Computes: down_proj @ (silu(gate_proj @ x) * (up_proj @ x))
///
/// `x`: [batch, seq_len, hidden_size]
/// `mlp`: MLP weight bundle, including optional Marlin W4A16-packed projections.
///
/// Dispatch each projection through the Marlin W4A16 path when the matching
/// `*_marlin` field is `Some`, else the existing BF16 `broadcast_matmul(*_t)`
/// path. LoRA deltas are always added on top so behaviour matches
/// `linear_with_lora_t` in the absence of Marlin weights. Mirrors
/// `q_proj_forward`'s Marlin routing from PR #149.
///
/// Returns: [batch, seq_len, hidden_size]
pub fn swiglu_ffn(
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    swiglu_ffn_impl(None, x, mlp, lora, false, None)
}

/// SwiGLU gate/up half used by exact training-time split backprop.
pub fn swiglu_ffn_gated_hidden(
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let gate = {
        kiln_nvtx::range!(c"kiln/mlp/gate");
        mlp_proj_forward_decode_if(
            None,
            false,
            x,
            &mlp.gate_proj_t,
            mlp.gate_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.gate_proj.as_ref()),
            lora_scale,
        )?
    };
    let up = {
        kiln_nvtx::range!(c"kiln/mlp/up");
        mlp_proj_forward_decode_if(
            None,
            false,
            x,
            &mlp.up_proj_t,
            mlp.up_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.up_proj.as_ref()),
            lora_scale,
        )?
    };
    #[cfg(feature = "metal")]
    {
        if crate::backend::metal::metal_mlp_silu_mul_supports(&gate, &up) {
            return crate::backend::metal::metal_mlp_silu_mul_bf16(&gate, &up)
                .context("metal mlp silu*mul kernel failed");
        }
    }
    #[cfg(feature = "cuda")]
    {
        if !cuda_fused_mlp_silu_mul_disabled()
            && !gate.track_op()
            && !up.track_op()
            && kiln_rmsnorm_kernel::supports_mlp_silu_mul(&gate, &up)
        {
            return kiln_rmsnorm_kernel::fused_mlp_silu_mul(&gate, &up)
                .context("cuda fused mlp silu*mul kernel failed");
        }
    }
    let gate = cuda_silu(&gate)?;
    (gate * up).map_err(Into::into)
}

/// SwiGLU down projection half used by exact training-time split backprop.
pub fn swiglu_ffn_down_from_gated(
    gated: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    mlp_proj_forward_decode_if(
        None,
        false,
        gated,
        &mlp.down_proj_t,
        mlp.down_proj_marlin.as_ref(),
        lora_layer.and_then(|l| l.down_proj.as_ref()),
        lora_scale,
    )
}

/// Transformer MLP gate/up half from a post-attention residual state.
pub fn transformer_mlp_gated_hidden(
    hidden: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let normed_post = {
        kiln_nvtx::range!(c"kiln/norm/pre_mlp");
        rms_norm(hidden, &layer.post_attention_layernorm, config.rms_norm_eps)?
    };
    swiglu_ffn_gated_hidden(&normed_post, &layer.mlp, lora)
}

/// Transformer MLP down half from a precomputed SwiGLU gated hidden.
pub fn transformer_mlp_down_from_gated(
    gated: &Tensor,
    layer: &GpuLayerWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    swiglu_ffn_down_from_gated(gated, &layer.mlp, lora)
}

fn swiglu_ffn_metal_decode(
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    swiglu_ffn_impl(None, x, mlp, lora, true, None)
}

fn swiglu_ffn_backend_profiled(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
    use_metal_decode_gemv: bool,
    profile_context: Option<(usize, usize)>,
) -> Result<Tensor> {
    swiglu_ffn_impl(
        Some(backend),
        x,
        mlp,
        lora,
        use_metal_decode_gemv,
        profile_context,
    )
}

fn swiglu_ffn_impl(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
    use_metal_decode_gemv: bool,
    profile_context: Option<(usize, usize)>,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    let (_, seq_len, _) = x.dims3()?;
    #[cfg(feature = "cuda")]
    {
        let chunk_tokens = cuda_training_mlp_chunk_tokens();
        if !cuda_training_mlp_chunking_disabled()
            && chunk_tokens > 0
            && seq_len > chunk_tokens
            && (x.track_op() || lora.is_some())
            && matches!(x.device(), Device::Cuda(_))
        {
            return swiglu_ffn_impl_chunked(
                backend,
                x,
                mlp,
                lora,
                use_metal_decode_gemv,
                profile_context,
                chunk_tokens,
            );
        }
    }

    swiglu_ffn_impl_no_chunk(
        backend,
        x,
        mlp,
        lora,
        use_metal_decode_gemv,
        profile_context,
    )
}

#[cfg(feature = "cuda")]
fn cuda_training_mlp_chunking_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| env_truthy("KILN_DISABLE_CUDA_TRAINING_MLP_CHUNKING"))
}

#[cfg(feature = "cuda")]
fn cuda_training_mlp_chunk_tokens() -> usize {
    static CHUNK_TOKENS: OnceLock<usize> = OnceLock::new();
    *CHUNK_TOKENS.get_or_init(|| {
        std::env::var("KILN_CUDA_TRAINING_MLP_CHUNK_TOKENS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(1024)
    })
}

#[cfg(feature = "cuda")]
fn swiglu_ffn_impl_chunked(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
    use_metal_decode_gemv: bool,
    profile_context: Option<(usize, usize)>,
    chunk_tokens: usize,
) -> Result<Tensor> {
    let (_, seq_len, _) = x.dims3()?;
    let mut outputs = Vec::with_capacity(seq_len.div_ceil(chunk_tokens));
    let mut start = 0usize;
    while start < seq_len {
        let len = (seq_len - start).min(chunk_tokens);
        let x_chunk = x.narrow(1, start, len).with_context(|| {
            format!(
                "chunked CUDA training MLP input tile [{start}, {})",
                start + len
            )
        })?;
        let out = swiglu_ffn_impl_no_chunk(
            backend,
            &x_chunk,
            mlp,
            lora,
            use_metal_decode_gemv,
            profile_context,
        )
        .with_context(|| format!("chunked CUDA training MLP tile [{start}, {})", start + len))?;
        outputs.push(out);
        start += len;
    }
    let output_refs: Vec<&Tensor> = outputs.iter().collect();
    Tensor::cat(&output_refs, 1).context("chunked CUDA training MLP cat")
}

fn swiglu_ffn_impl_no_chunk(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
    use_metal_decode_gemv: bool,
    profile_context: Option<(usize, usize)>,
) -> Result<Tensor> {
    let profile_device = x.device();
    let (_, seq_len, _) = x.dims3()?;
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let has_mlp_lora = lora_layer.is_some_and(LoraLayerWeights::has_mlp);
    let has_mlp_gate_up_lora = lora_layer.is_some_and(LoraLayerWeights::has_mlp_gate_up);
    let has_marlin = mlp.gate_proj_marlin.is_some()
        || mlp.up_proj_marlin.is_some()
        || mlp.down_proj_marlin.is_some();
    if !has_mlp_lora && !has_marlin {
        if let Some(backend) = backend {
            let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
            if let Some(out) =
                backend.mlp_decode(x, &mlp.gate_proj_t, &mlp.up_proj_t, &mlp.down_proj_t)?
            {
                finish_mlp_stage_profile(
                    profile_device,
                    profile_context,
                    "fused",
                    seq_len,
                    stage_profile,
                )?;
                return Ok(out);
            }
        }
    }
    if !has_mlp_gate_up_lora && !has_marlin {
        if let Some(backend) = backend {
            let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
            if let Some(hidden) = backend.mlp_gate_up_decode(x, &mlp.gate_proj_t, &mlp.up_proj_t)? {
                finish_mlp_stage_profile(
                    profile_device,
                    profile_context,
                    "gate_up_fused",
                    seq_len,
                    stage_profile,
                )?;
                let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
                let out = {
                    kiln_nvtx::range!(c"kiln/mlp/down");
                    mlp_proj_forward_decode_if(
                        Some(backend),
                        use_metal_decode_gemv,
                        &hidden,
                        &mlp.down_proj_t,
                        mlp.down_proj_marlin.as_ref(),
                        lora_layer.and_then(|l| l.down_proj.as_ref()),
                        lora_scale,
                    )?
                };
                finish_mlp_stage_profile(
                    profile_device,
                    profile_context,
                    "down_proj",
                    seq_len,
                    stage_profile,
                )?;
                return Ok(out);
            }
        }
    }
    #[cfg(feature = "metal")]
    let gate_up_profile = start_mlp_stage_profile(profile_device, profile_context)?;
    #[cfg(feature = "metal")]
    if let Some(hidden) = try_metal_mlp_gate_up_hidden(x, mlp, lora_layer)? {
        finish_mlp_stage_profile(
            profile_device,
            profile_context,
            "gate_up_fused",
            seq_len,
            gate_up_profile,
        )?;
        let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
        let out = {
            kiln_nvtx::range!(c"kiln/mlp/down");
            mlp_proj_forward_decode_if(
                backend,
                use_metal_decode_gemv,
                &hidden,
                &mlp.down_proj_t,
                mlp.down_proj_marlin.as_ref(),
                lora_layer.and_then(|l| l.down_proj.as_ref()),
                lora_scale,
            )?
        };
        finish_mlp_stage_profile(
            profile_device,
            profile_context,
            "down_proj",
            seq_len,
            stage_profile,
        )?;
        return Ok(out);
    }

    // x @ gate_proj_t -> [batch, seq_len, intermediate_size]
    let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
    let gate = {
        kiln_nvtx::range!(c"kiln/mlp/gate");
        mlp_proj_forward_decode_if(
            backend,
            use_metal_decode_gemv,
            x,
            &mlp.gate_proj_t,
            mlp.gate_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.gate_proj.as_ref()),
            lora_scale,
        )?
    };
    finish_mlp_stage_profile(
        profile_device,
        profile_context,
        "gate_proj",
        seq_len,
        stage_profile,
    )?;
    // x @ up_proj_t -> [batch, seq_len, intermediate_size]
    let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
    let up = {
        kiln_nvtx::range!(c"kiln/mlp/up");
        mlp_proj_forward_decode_if(
            backend,
            use_metal_decode_gemv,
            x,
            &mlp.up_proj_t,
            mlp.up_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.up_proj.as_ref()),
            lora_scale,
        )?
    };
    finish_mlp_stage_profile(
        profile_device,
        profile_context,
        "up_proj",
        seq_len,
        stage_profile,
    )?;
    let hidden = {
        #[cfg(feature = "metal")]
        {
            if crate::backend::metal::metal_mlp_silu_mul_supports(&gate, &up) {
                let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
                let hidden = crate::backend::metal::metal_mlp_silu_mul_bf16(&gate, &up)
                    .context("metal mlp silu*mul kernel failed")?;
                finish_mlp_stage_profile(
                    profile_device,
                    profile_context,
                    "gate_silu_hidden_mul",
                    seq_len,
                    stage_profile,
                )?;
                hidden
            } else {
                let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
                let gate = cuda_silu(&gate)?;
                finish_mlp_stage_profile(
                    profile_device,
                    profile_context,
                    "gate_silu",
                    seq_len,
                    stage_profile,
                )?;
                let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
                let hidden = (gate * up)?;
                finish_mlp_stage_profile(
                    profile_device,
                    profile_context,
                    "hidden_mul",
                    seq_len,
                    stage_profile,
                )?;
                hidden
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            #[cfg(feature = "cuda")]
            {
                if !cuda_fused_mlp_silu_mul_disabled()
                    && !gate.track_op()
                    && !up.track_op()
                    && kiln_rmsnorm_kernel::supports_mlp_silu_mul(&gate, &up)
                {
                    let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
                    let hidden = kiln_rmsnorm_kernel::fused_mlp_silu_mul(&gate, &up)
                        .context("cuda fused mlp silu*mul kernel failed")?;
                    finish_mlp_stage_profile(
                        profile_device,
                        profile_context,
                        "gate_silu_hidden_mul",
                        seq_len,
                        stage_profile,
                    )?;
                    hidden
                } else {
                    // SiLU activation: x * sigmoid(x)
                    let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
                    let gate = cuda_silu(&gate)?;
                    finish_mlp_stage_profile(
                        profile_device,
                        profile_context,
                        "gate_silu",
                        seq_len,
                        stage_profile,
                    )?;
                    // Element-wise multiply
                    let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
                    let hidden = (gate * up)?;
                    finish_mlp_stage_profile(
                        profile_device,
                        profile_context,
                        "hidden_mul",
                        seq_len,
                        stage_profile,
                    )?;
                    hidden
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                // SiLU activation: x * sigmoid(x)
                let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
                let gate = cuda_silu(&gate)?;
                finish_mlp_stage_profile(
                    profile_device,
                    profile_context,
                    "gate_silu",
                    seq_len,
                    stage_profile,
                )?;
                // Element-wise multiply
                let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
                let hidden = (gate * up)?;
                finish_mlp_stage_profile(
                    profile_device,
                    profile_context,
                    "hidden_mul",
                    seq_len,
                    stage_profile,
                )?;
                hidden
            }
        }
    };
    // hidden @ down_proj_t -> [batch, seq_len, hidden_size]
    let stage_profile = start_mlp_stage_profile(profile_device, profile_context)?;
    let out = {
        kiln_nvtx::range!(c"kiln/mlp/down");
        mlp_proj_forward_decode_if(
            backend,
            use_metal_decode_gemv,
            &hidden,
            &mlp.down_proj_t,
            mlp.down_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.down_proj.as_ref()),
            lora_scale,
        )?
    };
    finish_mlp_stage_profile(
        profile_device,
        profile_context,
        "down_proj",
        seq_len,
        stage_profile,
    )?;
    Ok(out)
}

/// Phase B12: sub-op-tapping variant of [`swiglu_ffn`]. Structurally
/// identical — same projections, same SiLU, same `gate * up` elementwise,
/// same down projection — but with three [`capture_b12_gqa_tap`] calls so
/// the HF comparator can localize drift to one of mlp_gate / mlp_up /
/// mlp_down on layer 31.
///
/// Called from [`transformer_block_paged`] only when
/// [`crate::mtp_debug::current_b12_layer_is_31`] is true, so the hot
/// production path continues to go through `swiglu_ffn` untouched.
fn swiglu_ffn_b12_tapped(
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    // mlp_gate: output of the gate projection BEFORE SiLU. This matches the
    // HF reference which taps `self.gate_proj(x)` pre-activation.
    let gate = {
        kiln_nvtx::range!(c"kiln/mlp/gate");
        mlp_proj_forward(
            x,
            &mlp.gate_proj_t,
            mlp.gate_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.gate_proj.as_ref()),
            lora_scale,
        )?
    };
    crate::mtp_debug::capture_b12_gqa_tap("mlp_gate", &gate)?;
    let gate = cuda_silu(&gate)?;
    // mlp_up: output of the up projection.
    let up = {
        kiln_nvtx::range!(c"kiln/mlp/up");
        mlp_proj_forward(
            x,
            &mlp.up_proj_t,
            mlp.up_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.up_proj.as_ref()),
            lora_scale,
        )?
    };
    crate::mtp_debug::capture_b12_gqa_tap("mlp_up", &up)?;
    let hidden = (gate * up)?;
    // mlp_down: final hidden-size output after the down projection.
    let out = {
        kiln_nvtx::range!(c"kiln/mlp/down");
        mlp_proj_forward(
            &hidden,
            &mlp.down_proj_t,
            mlp.down_proj_marlin.as_ref(),
            lora_layer.and_then(|l| l.down_proj.as_ref()),
            lora_scale,
        )?
    };
    crate::mtp_debug::capture_b12_gqa_tap("mlp_down", &out)?;
    Ok(out)
}

/// Route a single MLP projection through Marlin W4A16 when packed weights are
/// present, else fall back to the BF16 `linear_with_lora_t` path. LoRA deltas
/// are added on top of either base matmul. Mirrors `q_proj_forward`'s routing.
fn mlp_proj_forward(
    x: &Tensor,
    weight_t: &Tensor,
    marlin: Option<&crate::marlin_proj::MarlinPackedProj>,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    mlp_proj_forward_decode_if(None, false, x, weight_t, marlin, lora, lora_scale)
}

fn mlp_proj_forward_decode_if(
    backend: Option<&dyn BackendRuntime>,
    use_metal_decode_gemv: bool,
    x: &Tensor,
    weight_t: &Tensor,
    marlin: Option<&crate::marlin_proj::MarlinPackedProj>,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(packed) = marlin {
        let base = crate::marlin_proj::matmul_bf16(x, packed)
            .context("mlp_proj_forward: marlin matmul")?;
        if let Some(proj) = lora {
            let delta =
                compute_lora_delta(x, proj, lora_scale).context("mlp_proj_forward: lora delta")?;
            return Ok((base + delta).context("mlp_proj_forward: add lora delta")?);
        }
        return Ok(base);
    }
    // Non-CUDA builds never carry Marlin weights; reference the parameter so
    // the signature stays unified without a dead_code warning.
    let _ = marlin;
    linear_with_lora_t_backend_decode_if(
        backend,
        use_metal_decode_gemv,
        x,
        weight_t,
        lora,
        lora_scale,
    )
}

fn lm_head_forward(x: &Tensor, embed_tokens_t: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    {
        if crate::backend::metal::metal_lm_head_supports(x, embed_tokens_t) {
            return crate::backend::metal::metal_lm_head_bf16(x, embed_tokens_t)
                .context("metal lm_head kernel failed");
        }
        if crate::backend::metal::metal_transposed_coop_gemv_decode_batch_supports(
            x,
            embed_tokens_t,
        ) {
            return crate::backend::metal::metal_transposed_coop_gemv_bf16(x, embed_tokens_t)
                .context("metal batch lm_head GEMV failed");
        }
    }
    broadcast_matmul_cpu_compatible(x, embed_tokens_t)
}

fn lm_head_forward_backend_decode_if(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    embed_tokens_t: &Tensor,
) -> Result<Tensor> {
    if let Some(backend) = backend {
        // For autograd-tracked input (non-FLCE training path), prefer
        // the autograd-safe Vulkan CustomOp; otherwise the leaf
        // returned by linear_decode silently drops the gradient.
        if x.track_op() {
            if let Some(logits) = backend.linear_prefill_apply(x, embed_tokens_t)? {
                return Ok(logits);
            }
        }
        if let Some(logits) = backend.linear_decode(x, embed_tokens_t)? {
            return Ok(logits);
        }
    }
    lm_head_forward(x, embed_tokens_t)
}

fn lm_head_argmax(x: &Tensor, embed_tokens_t: &Tensor) -> Result<u32> {
    #[cfg(feature = "metal")]
    {
        if crate::backend::metal::metal_lm_head_argmax_supports(x, embed_tokens_t) {
            return crate::backend::metal::metal_lm_head_argmax_bf16(x, embed_tokens_t)
                .context("metal lm_head argmax kernel failed");
        }
    }
    let logits = lm_head_forward(x, embed_tokens_t)?;
    Ok(logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?)
}

fn lm_head_argmax_backend_decode_if(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    embed_tokens_t: &Tensor,
) -> Result<u32> {
    if let Some(backend) = backend {
        if let Some(token) = backend.linear_decode_argmax(x, embed_tokens_t)? {
            return Ok(token);
        }
    }
    lm_head_argmax(x, embed_tokens_t)
}

/// Token-history aggregation for the fused on-device sampling path.
/// Returns `(unique_indices, counts)` sorted by ascending token id so
/// the on-device scatter is deterministic across runs.
fn unique_history_counts(history: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut counts: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for &t in history {
        *counts.entry(t).or_default() += 1;
    }
    let mut idx = Vec::with_capacity(counts.len());
    let mut cnt = Vec::with_capacity(counts.len());
    for (k, v) in counts {
        idx.push(k);
        cnt.push(v);
    }
    (idx, cnt)
}

/// Hidden-state → sampled token, fully fused on-device when the backend
/// supports it. Returns `Ok(Some(token))` when the fused path ran;
/// `Ok(None)` when the backend declined (e.g. `top_k > kernel max`),
/// signalling to the caller that the legacy host sampler should run.
///
/// `params` is the full sampling spec (Qwen3.5-shaped). `step_seed` is
/// the per-step PRNG seed (overrides `params.seed` for this token);
/// `history` is the slice of generated tokens so far. Pass `&[]` for
/// the first decode token — penalties become a no-op under OpenAI
/// semantics.
#[allow(clippy::too_many_arguments)]
pub fn lm_head_sample_backend_decode_if(
    backend: Option<&dyn BackendRuntime>,
    hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    params: &kiln_core::sampling::SamplingParams,
    step_seed: Option<u64>,
    history: &[u32],
) -> Result<Option<u32>> {
    let Some(backend) = backend else {
        return Ok(None);
    };
    if !backend.supports_linear_decode_sample(params.top_k) {
        return Ok(None);
    }
    let normed = rms_norm(hidden, &weights.final_norm, config.rms_norm_eps)?;
    let (history_indices, history_counts) = unique_history_counts(history);
    let seed = step_seed.unwrap_or_else(|| {
        // PRNG seed for un-seeded requests — derived from nanos +
        // history hash so consecutive un-seeded tokens see distinct
        // entropy without burning a kernel side channel for it.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let h = history.iter().fold(0xCBF29CE484222325u64, |acc, &t| {
            (acc ^ t as u64).wrapping_mul(0x100000001B3)
        });
        nanos.wrapping_add(h)
    });
    backend.linear_decode_sample(
        &normed,
        &weights.embed_tokens_t,
        &history_indices,
        &history_counts,
        params.repetition_penalty,
        params.presence_penalty,
        params.frequency_penalty,
        params.temperature,
        params.top_k,
        params.top_p,
        params.min_p,
        seed,
    )
}

fn lm_head_argmax_rows(x: &Tensor, embed_tokens_t: &Tensor) -> Result<Vec<u32>> {
    #[cfg(feature = "metal")]
    {
        if crate::backend::metal::metal_lm_head_argmax_rows_supports(x, embed_tokens_t) {
            return crate::backend::metal::metal_lm_head_argmax_rows_bf16(x, embed_tokens_t)
                .context("metal batch lm_head argmax kernel failed");
        }
    }
    let logits = lm_head_forward(x, embed_tokens_t)?;
    crate::sampling::greedy_sample_rows(&logits).context("batched greedy row sampling failed")
}

fn lm_head_argmax_rows_backend_decode_if(
    backend: Option<&dyn BackendRuntime>,
    x: &Tensor,
    embed_tokens_t: &Tensor,
) -> Result<Vec<u32>> {
    if let Some(backend) = backend {
        if backend.supports_linear_decode_argmax_batch() {
            if let Some(tokens) = backend.linear_decode_argmax_batch(x, embed_tokens_t)? {
                return Ok(tokens);
            }
        }
    }
    lm_head_argmax_rows(x, embed_tokens_t)
}

fn lm_head_weighted_prep_argmax(
    x: &Tensor,
    norm_weight: &Tensor,
    embed_tokens_t: &Tensor,
) -> Result<Option<u32>> {
    if weighted_lm_head_prep_disabled()
        || x.dtype() != DType::BF16
        || norm_weight.dtype() != DType::BF16
        || !matches!(x.device(), Device::Metal(_))
        || !matches!(norm_weight.device(), Device::Metal(_))
    {
        return Ok(None);
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    if batch != 1 || seq_len != 1 || norm_weight.dims() != [hidden] {
        return Ok(None);
    }

    let weighted = x.broadcast_mul(norm_weight)?.contiguous()?;
    Ok(Some(lm_head_argmax(&weighted, embed_tokens_t)?))
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (GDN) linear attention primitives
// ---------------------------------------------------------------------------

/// L2 normalize the last dimension: x / sqrt(sum(x^2) + eps).
/// Returns result in F32 regardless of input dtype.
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq_sum = x_f32.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
    let norm = (sq_sum + 1e-6)?.sqrt()?;
    let normalized = x_f32.broadcast_div(&norm)?;
    Ok(normalized)
}

fn gdn_qk_norm(q: &Tensor, k: &Tensor, input_dtype: DType, scale: f64) -> Result<(Tensor, Tensor)> {
    #[cfg(any(feature = "metal", feature = "cuda"))]
    let fused_forward_only_allowed = !any_tensor_tracks_op(&[q, k]);
    #[cfg(feature = "metal")]
    {
        if fused_forward_only_allowed
            && input_dtype == DType::BF16
            && crate::backend::metal::metal_gdn_qk_norm_supports(q, k)
        {
            return crate::backend::metal::metal_gdn_qk_norm_f32_bf16(q, k, scale as f32, 1e-6)
                .context("metal gdn qk_norm kernel failed");
        }
    }

    #[cfg(feature = "cuda")]
    {
        let disabled = std::env::var("KILN_DISABLE_FUSED_L2_QK_NORM").is_ok();
        if !disabled
            && fused_forward_only_allowed
            && input_dtype == DType::BF16
            && kiln_rmsnorm_kernel::supports_l2_qk_norm(q, k)
        {
            return kiln_rmsnorm_kernel::fused_l2_qk_norm(q, k, scale as f32, 1e-6)
                .context("fused_l2_qk_norm kernel failed");
        }
    }

    let q = l2_normalize(q)?; // F32
    let k = l2_normalize(k)?; // F32
    let q = (q * scale)?.to_dtype(input_dtype)?;
    let k = k.to_dtype(input_dtype)?;
    Ok((q, k))
}

/// softplus(x) = ln(1 + exp(x)), numerically stable for all x.
///
/// Uses the identity: softplus(x) = max(x, 0) + ln(1 + exp(-|x|))
/// Since exp(-|x|) ∈ (0, 1], no overflow is possible.
/// This matches PyTorch's F.softplus output (which clamps to linear for x > 20).
fn softplus(x: &Tensor) -> Result<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    let relu_x = x.maximum(&zeros)?;
    // |x| = relu(x) + relu(-x)
    let neg_x = x.neg()?;
    let relu_neg_x = neg_x.maximum(&zeros)?;
    let abs_x = (relu_x.clone() + relu_neg_x)?;
    let neg_abs = abs_x.neg()?;
    // log(1 + exp(-|x|)) — always stable since exp(-|x|) ∈ (0, 1]
    let log_term = (neg_abs.exp()? + 1.0)?.log()?;
    Ok((relu_x + log_term)?)
}

fn gated_rms_norm(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    if !any_tensor_tracks_op(&[x, z, weight]) && backend.supports_gdn_gated_rms_norm() {
        if let Some(out) = backend.gdn_gated_rms_norm(x, z, weight, eps)? {
            return Ok(out);
        }
    }
    gated_rms_norm_fallback(x, z, weight, eps)
}

/// Gated RMSNorm: rms_norm(x, weight) * silu(z).
///
/// Applied per-group on the last dimension. Returns F32.
///
/// `x`: [..., dim] — attention output
/// `z`: [..., dim] — output gate (from in_proj_z)
/// `weight`: [dim] — learnable scale
fn gated_rms_norm_fallback(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let z_f32 = z.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?;

    // RMS norm on last dimension
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rms_inv = (variance + eps)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&rms_inv)?;
    let normed = normed.broadcast_mul(&w_f32)?;

    // Output gate: silu(z) = z * sigmoid(z)
    let gate = cuda_silu(&z_f32)?;
    let out = (normed * gate)?;
    Ok(out)
}

/// Causal depthwise conv1d for prefill (seq_len > 1).
///
/// `x`: [batch, channels, seq_len]
/// `weight`: [channels, 1, kernel_size]
/// `conv_state`: [batch, channels, kernel_size - 1] — updated to last K-1 inputs
fn causal_conv1d_prefill(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    let compute_dtype = causal_conv1d_prefill_compute_dtype(x, weight, conv_state, kernel_size);
    causal_conv1d_prefill_with_dtype(x, weight, conv_state, kernel_size, compute_dtype)
}

fn causal_conv1d_prefill_compute_dtype(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
) -> DType {
    if matches!(x.device(), Device::Metal(_))
        && x.dtype() == DType::BF16
        && weight.dtype() == DType::BF16
        && conv_state.dtype() == DType::F32
        && kernel_size == 4
    {
        DType::BF16
    } else {
        DType::F32
    }
}

fn causal_conv1d_prefill_with_dtype(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
    compute_dtype: DType,
) -> Result<Tensor> {
    let (_batch, channels, seq_len) = x.dims3()?;
    let x_compute = x.to_dtype(compute_dtype)?;
    let x_state_f32 = if compute_dtype == DType::F32 {
        x_compute.clone()
    } else {
        x.to_dtype(DType::F32)?
    };
    // Squeeze [channels, 1, kernel_size] -> [channels, kernel_size]
    let w_compute = weight
        .to_dtype(compute_dtype)?
        .reshape((channels, kernel_size))?;
    let k_minus_1 = kernel_size - 1;

    // Left-pad with conv_state (previous K-1 inputs, or zeros for fresh state)
    let x_padded = Tensor::cat(&[&conv_state.to_dtype(compute_dtype)?, &x_compute], 2)?;

    // Depthwise conv: output[t] = sum_{j=0}^{K-1} weight[j] * x_padded[t+j]
    let mut output = Tensor::zeros_like(&x_compute)?;
    for j in 0..kernel_size {
        let x_slice = x_padded.narrow(2, j, seq_len)?;
        let w_j = w_compute.narrow(1, j, 1)?.unsqueeze(0)?; // [1, channels, 1]
        output = (output + x_slice.broadcast_mul(&w_j)?)?;
    }

    // Update conv_state to the last K-1 input positions
    if seq_len >= k_minus_1 {
        *conv_state = x_state_f32
            .narrow(2, seq_len - k_minus_1, k_minus_1)?
            .contiguous()?;
    } else {
        // Fewer new tokens than buffer size: shift old state and append new
        let keep = k_minus_1 - seq_len;
        let old_part = conv_state.narrow(2, seq_len, keep)?;
        *conv_state = Tensor::cat(&[&old_part, &x_state_f32], 2)?.contiguous()?;
    }

    Ok(output)
}

/// Causal depthwise conv1d for decode (seq_len == 1).
///
/// `x`: [batch, channels, 1]
/// `weight`: [channels, 1, kernel_size]
/// `conv_state`: [batch, channels, kernel_size - 1] — updated
fn causal_conv1d_decode(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    let (_batch, channels, _one) = x.dims3()?;
    let x_f32 = x.to_dtype(DType::F32)?;
    let w_f32 = weight
        .to_dtype(DType::F32)?
        .reshape((channels, kernel_size))?;

    // Full window = [conv_state | x] -> [batch, channels, kernel_size]
    let window = Tensor::cat(&[&conv_state.to_dtype(DType::F32)?, &x_f32], 2)?;

    // Dot product per channel: sum over kernel dimension
    let w_expanded = w_f32.unsqueeze(0)?; // [1, channels, kernel_size]
    let output = window.broadcast_mul(&w_expanded)?.sum(2)?; // [batch, channels]
    let output = output.unsqueeze(2)?; // [batch, channels, 1]

    // Update conv_state in place: drop oldest, append newest. CUDA graph
    // capture bakes the conv_state device pointer into later decode kernels, so
    // rebinding `conv_state` to a newly allocated tensor during capture leaves
    // replay with a dangling pointer. Keep the caller-owned storage stable.
    let next_state = window.narrow(2, 1, kernel_size - 1)?.contiguous()?;
    conv_state
        .slice_set(&next_state, 0, 0)
        .context("update decode conv_state in place")?;

    Ok(output)
}

// ---------------------------------------------------------------------------
// GDN chunkwise analytical recurrence (Phase 6, approach (b) in the chunkwise
// plan). Replaces the per-token `for t in 0..seq_len` loop inside
// `gated_deltanet_forward` with an unrolled form that processes up to
// `GDN_CHUNK_SIZE` tokens per heavy matmul, dropping the number of GPU kernel
// launches from O(T) to O(T / C) per layer.
// ---------------------------------------------------------------------------

/// Chunk size for the analytical GDN recurrence. C = 64 balances:
///   - intra-chunk [C, dk] × [dk, C] matmuls large enough to saturate tensor
///     cores on A5000/4090-class GPUs for dk = dv = 128,
///   - a small-enough forward-substitution inner loop so the Vec<Tensor> cat
///     churn stays bounded.
pub const GDN_CHUNK_SIZE: usize = 64;
const GDN_RECURRENT_PREFILL_MAX_TOKENS: usize = 2048;

/// Build a [n, n] mask on `device` with `dtype`, 1.0 where row > col else 0.0.
/// Used for the strictly lower-triangular `A_strict` mask (i < t, exclusive).
fn strict_lower_tri_bool(n: usize, device: &Device) -> Result<Tensor> {
    let t = Tensor::arange(0u32, n as u32, device)?;
    let cols = t.reshape((1, n))?.broadcast_as((n, n))?;
    let rows = t.reshape((n, 1))?.broadcast_as((n, n))?;
    Ok(rows.gt(&cols)?)
}

#[cfg(test)]
#[allow(dead_code)]
fn strict_lower_tri_mask(n: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    Ok(strict_lower_tri_bool(n, device)?.to_dtype(dtype)?)
}

/// Build a [n, n] mask on `device` with `dtype`, 1.0 where row >= col else 0.0.
/// Used for the causal (inclusive) lower-triangular `B_mask` mask (i <= t).
fn causal_lower_tri_bool(n: usize, device: &Device) -> Result<Tensor> {
    let t = Tensor::arange(0u32, n as u32, device)?;
    let cols = t.reshape((1, n))?.broadcast_as((n, n))?;
    let rows = t.reshape((n, 1))?.broadcast_as((n, n))?;
    Ok(rows.ge(&cols)?)
}

#[cfg(test)]
#[allow(dead_code)]
fn causal_lower_tri_mask(n: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    Ok(causal_lower_tri_bool(n, device)?.to_dtype(dtype)?)
}

/// Compute the chunk-local W = (I + A_strict)^{-1} (beta * V_prime) by
/// forward substitution. On backends that advertise
/// `supports_gdn_forward_substitution()` (CUDA/Metal bf16 today), dispatches
/// to the fused kernel (one kernel block per (batch, head)) when
/// `chunk_size <= 128`. Otherwise it falls back to the per-token candle loop.
fn compute_w_chunk(
    backend: &dyn BackendRuntime,
    a_strict: &Tensor, // [B, nv, C, C]
    v_prime: &Tensor,  // [B, nv, C, dv]
    beta_c: &Tensor,   // [B, nv, C]
    c: usize,
) -> Result<Tensor> {
    // The kernel envelope is C <= 128; callers enforce this precondition so
    // we never pay for a backend call we know will decline.
    if c <= 128
        && !any_tensor_tracks_op(&[a_strict, v_prime, beta_c])
        && backend.supports_gdn_forward_substitution()
    {
        kiln_nvtx::range!(c"kiln/attn/gdn/chunk");
        if let Some(out) = backend.gdn_forward_substitution(a_strict, v_prime, beta_c)? {
            return Ok(out);
        }
    }
    compute_w_chunk_fallback(a_strict, v_prime, beta_c, c)
}

/// Reference per-token forward substitution. Kept as the CPU path and as
/// the correctness oracle for the fused CUDA kernel.
fn compute_w_chunk_fallback(
    a_strict: &Tensor,
    v_prime: &Tensor,
    beta_c: &Tensor,
    c: usize,
) -> Result<Tensor> {
    let beta_col = beta_c.unsqueeze(3)?; // [B, nv, C, 1]
    let mut w_rows: Vec<Tensor> = Vec::with_capacity(c);
    for t in 0..c {
        let vp_t = v_prime.narrow(2, t, 1)?; // [B, nv, 1, dv]
        let beta_t = beta_col.narrow(2, t, 1)?; // [B, nv, 1, 1]
        let w_t = if t == 0 {
            vp_t.broadcast_mul(&beta_t)?
        } else {
            let a_row = a_strict.narrow(2, t, 1)?.narrow(3, 0, t)?.contiguous()?;
            let w_prev = Tensor::cat(&w_rows, 2)?;
            let sub = a_row.matmul(&w_prev)?; // [B, nv, 1, dv]
            (vp_t - sub)?.broadcast_mul(&beta_t)?
        };
        w_rows.push(w_t);
    }
    Ok(Tensor::cat(&w_rows, 2)?)
}

#[allow(dead_code)]
fn compute_chunk_body_reference(
    a_strict: &Tensor,
    b_mask: &Tensor,
    v_prime: &Tensor,
    q_s_scaled: &Tensor,
    beta_c: &Tensor,
    decay_last_col_u: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let c = v_prime.dim(2)?;
    let w = compute_w_chunk_fallback(a_strict, v_prime, beta_c, c)?;
    let intra = b_mask.matmul(&w)?;
    let out_chunk = (q_s_scaled + &intra)?;
    let w_weighted = w.broadcast_mul(decay_last_col_u)?.contiguous()?;
    Ok((out_chunk, w_weighted))
}

/// Specialized single-token GDN recurrence.
///
/// This is the non-CUDA fast path for `seq_len == 1`, avoiding the chunkwise
/// prep work (`KKT`, `QKT`, masks, triangular solve) that is only worthwhile
/// when a chunk contains multiple tokens.
fn gdn_single_token_recurrence(
    q: &Tensor,         // [B, nv, 1, dk]
    k: &Tensor,         // [B, nv, 1, dk]
    v: &Tensor,         // [B, nv, 1, dv]
    beta: &Tensor,      // [B, nv, 1]
    g: &Tensor,         // [B, nv, 1]
    state: &mut Tensor, // [B, nv, dk, dv]
) -> Result<Tensor> {
    let dtype = q.dtype();

    let p = g.to_dtype(DType::F32)?.exp()?.to_dtype(dtype)?;
    let p_u = p.unsqueeze(3)?; // [B, nv, 1, 1]

    let k_t = k.transpose(2, 3)?.contiguous()?; // [B, nv, dk, 1]
    let ks_entry = k.matmul(&*state)?; // [B, nv, 1, dv]
    let q_s = q.matmul(&*state)?; // [B, nv, 1, dv]

    let v_prime = (v - ks_entry.broadcast_mul(&p_u)?)?;
    let w = v_prime.broadcast_mul(&beta.unsqueeze(3)?)?; // [B, nv, 1, dv]
    let qk = q.matmul(&k_t)?; // [B, nv, 1, 1]
    let out = (q_s.broadcast_mul(&p_u)? + qk.matmul(&w)?)?;

    let state_scaled = state.broadcast_mul(&p_u)?;
    let delta_state = k_t.matmul(&w)?;
    *state = (state_scaled + delta_state)?;

    Ok(out)
}

/// Analytical chunkwise form of the Gated DeltaNet recurrence.
///
/// The per-token recurrence is
///
/// ```text
///   S_t   = exp(g_t) * S_{t-1}  +  k_t ⊗ delta_t
///   delta_t = beta_t * (v_t - k_t · (exp(g_t) * S_{t-1}))
///   out_t = q_t · S_t
/// ```
///
/// Within a chunk of up to `chunk_size` tokens, let `G[t] = cumsum(g)[t]`.
/// The per-token recurrence unrolls into the closed form (derived from the
/// standard GLA / chunk_gla_fwd identity used in fla-org and RWKV-5):
///
/// 1. Inter-chunk carry
///    ```text
///      V'[t] = v[t] - exp(G[t]) * (k[t] · S_entry)
///    ```
/// 2. Strict intra-chunk decay mask
///    ```text
///      A_strict[t, i] = exp(G[t] - G[i]) * (k[t] · k[i])   for i < t, else 0
///    ```
/// 3. Forward-substitution / triangular solve for W[t]
///    ```text
///      W[t] = beta[t] * ( V'[t] - Σ_{i<t} A_strict[t, i] * W[i] )
///    ```
/// 4. Output
///    ```text
///      B_mask[t, i] = exp(G[t] - G[i]) * (q[t] · k[i])     for i <= t, else 0
///      out[t] = exp(G[t]) * (q[t] · S_entry) + Σ_{i<=t} B_mask[t, i] * W[i]
///    ```
/// 5. State exit
///    ```text
///      S_new = exp(G[C-1]) * S_entry + Σ_i exp(G[C-1] - G[i]) * k[i] ⊗ W[i]
///    ```
///
/// This is numerically equivalent to the per-token loop (modulo rounding in
/// the bf16 hot path) and matches the pre-existing sequential code exactly
/// for chunk_size = 1 (decode path).
///
/// Inputs are already transposed to `[B, nv, T, *]` layout. `state` is
/// mutated in place and must be in the hot-path dtype (bf16 in production,
/// F32 on CPU tests); the caller is responsible for preserving the external
/// F32-state invariant.
///
/// Returns: `[B, nv, T, dv]`.
fn gdn_chunkwise_recurrence(
    backend: &dyn BackendRuntime,
    q: &Tensor,         // [B, nv, T, dk]
    k: &Tensor,         // [B, nv, T, dk]
    v: &Tensor,         // [B, nv, T, dv]
    beta: &Tensor,      // [B, nv, T]
    g: &Tensor,         // [B, nv, T]
    state: &mut Tensor, // [B, nv, dk, dv]
    chunk_size: usize,
) -> Result<Tensor> {
    let (batch, heads, seq_len, _) = q.dims4()?;
    let dtype = q.dtype();
    let device = q.device();
    let profile_inner = profile_gdn_recurrent_inner_stages_enabled();

    // Single-token decode fast path. The chunkwise machinery (preshape,
    // decay matrix, KKT, forward sub, B_mask) costs more than the per-token
    // recurrence itself when seq_len == 1, which is the cause of the −54%
    // decode regression in PR #80. The backend's `gdn_recurrent_step`
    // kernel (CUDA today) collapses the whole recurrence into one block
    // per (B,H).
    if seq_len == 1 {
        let use_backend_recurrent_step = state.dtype() == dtype
            && !any_tensor_tracks_op(&[q, k, v, beta, g, state])
            && backend.supports_gdn_recurrent_step()
            && (dtype == DType::BF16
                || (dtype == DType::F32
                    && backend.name() == "vulkan"
                    && vulkan_gdn_recurrent_step_f32_enabled()));
        if use_backend_recurrent_step {
            // The five squeeze+contiguous calls below can copy the single-row
            // inputs before the recurrent forward runs. The dedicated NVTX
            // range lets nsys attribute this separately from the kernel itself.
            let stage_profile = start_gdn_recurrent_inner_profile(device, profile_inner)?;
            let (q1, k1, v1, beta1, g1) = {
                kiln_nvtx::range!(c"kiln/attn/gdn/precopy");
                (
                    q.squeeze(2)?.contiguous()?,
                    k.squeeze(2)?.contiguous()?,
                    v.squeeze(2)?.contiguous()?,
                    beta.squeeze(2)?.contiguous()?,
                    g.squeeze(2)?.contiguous()?,
                )
            };
            finish_gdn_recurrent_inner_profile(
                device,
                "single_token_precopy",
                batch,
                heads,
                seq_len,
                0,
                1,
                stage_profile,
            )?;
            let stage_profile = start_gdn_recurrent_inner_profile(device, profile_inner)?;
            let out_opt = {
                kiln_nvtx::range!(c"kiln/attn/gdn/recurrent");
                backend.gdn_recurrent_step(&q1, &k1, &v1, &beta1, &g1, state)?
            };
            finish_gdn_recurrent_inner_profile(
                device,
                "single_token_backend_step",
                batch,
                heads,
                seq_len,
                0,
                1,
                stage_profile,
            )?;
            if let Some(out) = out_opt {
                return Ok(out.unsqueeze(2)?);
            }
        }

        let stage_profile = start_gdn_recurrent_inner_profile(device, profile_inner)?;
        let out = gdn_single_token_recurrence(q, k, v, beta, g, state)?;
        finish_gdn_recurrent_inner_profile(
            device,
            "single_token_fallback",
            batch,
            heads,
            seq_len,
            0,
            1,
            stage_profile,
        )?;
        return Ok(out);
    }

    let full_chunks = seq_len / chunk_size;
    let tail = seq_len - full_chunks * chunk_size;

    // Slice full chunks directly. On macOS Metal this avoids the large upfront
    // pre-permute copies that dominated long-prompt GDN recurrence time.

    let mut out_chunks: Vec<Tensor> = Vec::with_capacity(seq_len.div_ceil(chunk_size));

    for ci in 0..(full_chunks + if tail > 0 { 1 } else { 0 }) {
        let is_tail = ci >= full_chunks;
        let c = if is_tail { tail } else { chunk_size };

        let stage_profile = start_gdn_recurrent_inner_profile(device, profile_inner)?;
        let (q_c, k_c, v_c, beta_c, g_c) = if is_tail {
            let t_start = full_chunks * chunk_size;
            (
                q.narrow(2, t_start, tail)?.contiguous()?,
                k.narrow(2, t_start, tail)?.contiguous()?,
                v.narrow(2, t_start, tail)?.contiguous()?,
                beta.narrow(2, t_start, tail)?.contiguous()?,
                g.narrow(2, t_start, tail)?.contiguous()?,
            )
        } else {
            let t_start = ci * chunk_size;
            (
                q.narrow(2, t_start, chunk_size)?.contiguous()?,
                k.narrow(2, t_start, chunk_size)?.contiguous()?,
                v.narrow(2, t_start, chunk_size)?.contiguous()?,
                beta.narrow(2, t_start, chunk_size)?.contiguous()?,
                g.narrow(2, t_start, chunk_size)?.contiguous()?,
            )
        };
        finish_gdn_recurrent_inner_profile(
            device,
            "slice_inputs",
            batch,
            heads,
            seq_len,
            ci,
            c,
            stage_profile,
        )?;

        // Matmuls first — these are well-tuned cuBLAS GEMMs and stay on
        // candle. K^T is reused for KKT (intra-chunk similarities) and the
        // final outer product into the state update.
        let stage_profile = start_gdn_recurrent_inner_profile(device, profile_inner)?;
        let k_t_mat = k_c.transpose(2, 3)?.contiguous()?; // [B, nv, dk, C]
        let ks_entry = k_c.matmul(&*state)?; // [B, nv, C, dv]
        let kkt = k_c.matmul(&k_t_mat)?; // [B, nv, C, C]
        let qkt = q_c.matmul(&k_t_mat)?; // [B, nv, C, C]
        let q_s = q_c.matmul(&*state)?; // [B, nv, C, dv]
        finish_gdn_recurrent_inner_profile(
            device,
            "matmul_prep",
            batch,
            heads,
            seq_len,
            ci,
            c,
            stage_profile,
        )?;

        if !is_tail
            && c == 64
            && !any_tensor_tracks_op(&[
                &g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s, &beta_c, &k_t_mat, state,
            ])
            && backend.supports_gdn_full_chunk_forward()
            && dtype == DType::BF16
        {
            let stage_profile = start_gdn_recurrent_inner_profile(device, profile_inner)?;
            let out_chunk = backend.gdn_full_chunk_forward(
                &g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s, &beta_c, &k_t_mat, state,
            )?;
            finish_gdn_recurrent_inner_profile(
                device,
                "full_chunk_forward",
                batch,
                heads,
                seq_len,
                ci,
                c,
                stage_profile,
            )?;
            if let Some(out_chunk) = out_chunk {
                out_chunks.push(out_chunk);
                continue;
            }
        }

        // Fused prep: cumsum + decay + exp + masked scales + v_prime +
        // q_s_scaled + decay_last_col + p_last in a single CUDA launch.
        // Falls back to the candle-op chain when the backend declines
        // (non-CUDA, non-bf16, envelope violation).
        //
        // Post-conditions on all four paths:
        //   a_strict:         [B, nv, C, C] bf16 — kkt * decay * strict_lower
        //   b_mask:           [B, nv, C, C] bf16 — qkt * decay * causal_lower
        //   v_prime:          [B, nv, C, dv] bf16 — v - ks_entry * p
        //   q_s_scaled:       [B, nv, C, dv] bf16 — q_s * p
        //   decay_last_col_u: [B, nv, C, 1]  bf16 — exp(big_g[C-1] - big_g[i])
        //   p_last_u:         [B, nv, 1, 1]  bf16 — exp(big_g[C-1])
        let stage_profile = start_gdn_recurrent_inner_profile(device, profile_inner)?;
        let (a_strict, b_mask, v_prime, q_s_scaled, decay_last_col_u, p_last_u) = {
            kiln_nvtx::range!(c"kiln/attn/gdn/chunk_prep");
            let prep_out = if !any_tensor_tracks_op(&[&g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s])
                && backend.supports_gdn_chunk_prep()
                && dtype == DType::BF16
            {
                backend.gdn_chunk_prep(&g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s)?
            } else {
                None
            };
            match prep_out {
                Some((a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last)) => {
                    let decay_last_col_u = decay_last_col.unsqueeze(3)?; // [B,nv,C,1]
                    let p_last_u = p_last.unsqueeze(2)?.unsqueeze(3)?; // [B,nv,1,1]
                    (
                        a_strict.contiguous()?,
                        b_mask.contiguous()?,
                        v_prime,
                        q_s_scaled,
                        decay_last_col_u,
                        p_last_u,
                    )
                }
                None => {
                    // Cumulative decay G[t] = Σ_{s=0..t} g[s].  Done in F32:
                    // exp() of the cumulative sum is the only place bf16
                    // would lose meaningful precision (G can reach -10 or
                    // more across a full 64-token chunk).
                    let g_f32 = g_c.to_dtype(DType::F32)?;
                    let big_g = g_f32.cumsum(candle_core::D::Minus1)?; // [B, nv, C], F32

                    // Decay matrix D[t, i] = exp(G[t] - G[i]). Mask before
                    // exp: masked future positions can otherwise overflow to
                    // inf, and inf * 0 is NaN.
                    let big_g_col = big_g.unsqueeze(3)?; // [B, nv, C, 1]
                    let big_g_row = big_g.unsqueeze(2)?; // [B, nv, 1, C]
                    let decay_delta = big_g_col.broadcast_sub(&big_g_row)?;
                    let zero_delta = Tensor::zeros_like(&decay_delta)?;
                    let strict_bool = strict_lower_tri_bool(c, device)?
                        .reshape((1, 1, c, c))?
                        .broadcast_as((batch, heads, c, c))?;
                    let causal_bool = causal_lower_tri_bool(c, device)?
                        .reshape((1, 1, c, c))?
                        .broadcast_as((batch, heads, c, c))?;
                    let strict_decay = strict_bool
                        .where_cond(&decay_delta, &zero_delta)?
                        .exp()?
                        .to_dtype(dtype)?;
                    let causal_decay = causal_bool
                        .where_cond(&decay_delta, &zero_delta)?
                        .exp()?
                        .to_dtype(dtype)?;

                    // p[t] = exp(G[t]).
                    let p = big_g.exp()?.to_dtype(dtype)?; // [B, nv, C]
                    let p_col = p.unsqueeze(3)?; // [B, nv, C, 1]

                    let strict_mask = strict_bool.to_dtype(dtype)?;
                    let causal_mask = causal_bool.to_dtype(dtype)?;

                    let v_prime = (&v_c - ks_entry.broadcast_mul(&p_col)?)?;
                    let a_strict = kkt
                        .broadcast_mul(&strict_decay)?
                        .broadcast_mul(&strict_mask)?
                        .contiguous()?;
                    let b_mask = qkt
                        .broadcast_mul(&causal_decay)?
                        .broadcast_mul(&causal_mask)?
                        .contiguous()?;
                    let q_s_scaled = q_s.broadcast_mul(&p_col)?;

                    let g_last = big_g.narrow(2, c - 1, 1)?; // [B, nv, 1]
                    let decay_last_col_u = g_last
                        .broadcast_sub(&big_g)?
                        .exp()?
                        .to_dtype(dtype)?
                        .unsqueeze(3)?; // [B, nv, C, 1]
                    let p_last_u = g_last.exp()?.to_dtype(dtype)?.unsqueeze(3)?; // [B,nv,1,1]

                    (
                        a_strict,
                        b_mask,
                        v_prime,
                        q_s_scaled,
                        decay_last_col_u,
                        p_last_u,
                    )
                }
            }
        };
        finish_gdn_recurrent_inner_profile(
            device,
            "chunk_prep",
            batch,
            heads,
            seq_len,
            ci,
            c,
            stage_profile,
        )?;

        let decay_last_col = decay_last_col_u.squeeze(3)?;
        let stage_profile = start_gdn_recurrent_inner_profile(device, profile_inner)?;
        let (out_chunk, w_weighted) = {
            kiln_nvtx::range!(c"kiln/attn/gdn/chunk");
            if !any_tensor_tracks_op(&[
                &a_strict,
                &b_mask,
                &v_prime,
                &q_s_scaled,
                &beta_c,
                &decay_last_col,
            ]) && backend.supports_gdn_chunk_scan()
                && dtype == DType::BF16
            {
                match backend.gdn_chunk_scan(
                    &a_strict,
                    &b_mask,
                    &v_prime,
                    &q_s_scaled,
                    &beta_c,
                    &decay_last_col,
                )? {
                    Some((out_chunk, w_weighted)) => (out_chunk, w_weighted),
                    None => {
                        let w = compute_w_chunk(backend, &a_strict, &v_prime, &beta_c, c)?;
                        let intra = b_mask.matmul(&w)?;
                        let out_chunk = (&q_s_scaled + &intra)?;
                        let w_weighted = w.broadcast_mul(&decay_last_col_u)?.contiguous()?;
                        (out_chunk, w_weighted)
                    }
                }
            } else {
                let w = compute_w_chunk(backend, &a_strict, &v_prime, &beta_c, c)?;
                let intra = b_mask.matmul(&w)?;
                let out_chunk = (&q_s_scaled + &intra)?;
                let w_weighted = w.broadcast_mul(&decay_last_col_u)?.contiguous()?;
                (out_chunk, w_weighted)
            }
        };
        finish_gdn_recurrent_inner_profile(
            device,
            "chunk_scan",
            batch,
            heads,
            seq_len,
            ci,
            c,
            stage_profile,
        )?;

        out_chunks.push(out_chunk); // [B, nv, C, dv]

        // State update:
        //   S_new = exp(G[C-1]) * S_entry
        //         + Σ_i exp(G[C-1] - G[i]) * k[i] ⊗ W[i]
        let stage_profile = start_gdn_recurrent_inner_profile(device, profile_inner)?;
        let state_scaled = state.broadcast_mul(&p_last_u)?; // [B, nv, dk, dv]
        let delta_state = k_t_mat.matmul(&w_weighted)?; // [B, nv, dk, dv]
        *state = (state_scaled + delta_state)?;
        finish_gdn_recurrent_inner_profile(
            device,
            "state_update",
            batch,
            heads,
            seq_len,
            ci,
            c,
            stage_profile,
        )?;
    }

    let stage_profile = start_gdn_recurrent_inner_profile(device, profile_inner)?;
    let out = Tensor::cat(&out_chunks, 2)?;
    finish_gdn_recurrent_inner_profile(
        device,
        "cat_out",
        batch,
        heads,
        seq_len,
        full_chunks + if tail > 0 { 1 } else { 0 },
        seq_len,
        stage_profile,
    )?;
    Ok(out)
}

fn gdn_recurrent_prefill_head_last(
    backend: &dyn BackendRuntime,
    q: &Tensor,         // [B, nv, T, dk]
    k: &Tensor,         // [B, nv, T, dk]
    v: &Tensor,         // [B, nv, T, dv]
    beta: &Tensor,      // [B, nv, T]
    g: &Tensor,         // [B, nv, T]
    state: &mut Tensor, // [B, nv, dk, dv]
) -> Result<Option<Tensor>> {
    let (_, _, seq_len, _) = q.dims4()?;
    if seq_len <= 1
        || q.dtype() != DType::BF16
        || state.dtype() != DType::BF16
        || any_tensor_tracks_op(&[q, k, v, beta, g, state])
        || !backend.supports_gdn_recurrent_prefill_head_last()
    {
        return Ok(None);
    }
    backend.gdn_recurrent_prefill_head_last(q, k, v, beta, g, state)
}

fn gdn_recurrent_prefill_native_head_last(
    backend: &dyn BackendRuntime,
    q: &Tensor,         // [B, T, nk, dk]
    k: &Tensor,         // [B, T, nk, dk]
    v: &Tensor,         // [B, T, nv, dv]
    beta: &Tensor,      // [B, T, nv]
    g: &Tensor,         // [B, T, nv]
    state: &mut Tensor, // [B, nv, dk, dv]
) -> Result<Option<Tensor>> {
    let (_, seq_len, _, _) = q.dims4()?;
    if seq_len == 0
        || q.dtype() != DType::BF16
        || state.dtype() != DType::BF16
        || any_tensor_tracks_op(&[q, k, v, beta, g, state])
        || !backend.supports_gdn_recurrent_prefill_native_head_last()
    {
        return Ok(None);
    }
    backend.gdn_recurrent_prefill_native_head_last(q, k, v, beta, g, state)
}

/// Metal BF16 fast path for full 64-token chunks.
///
/// Returns a contiguous head-last `[B, T, nv, dv]` tensor so the caller can feed
/// Metal gated RMSNorm without the `[B,nv,T,dv]` cat + transpose + contiguous
/// copy chain.
fn gdn_chunkwise_recurrence_head_last_full_chunks(
    backend: &dyn BackendRuntime,
    q: &Tensor,         // [B, nv, T, dk]
    k: &Tensor,         // [B, nv, T, dk]
    v: &Tensor,         // [B, nv, T, dv]
    beta: &Tensor,      // [B, nv, T]
    g: &Tensor,         // [B, nv, T]
    state: &mut Tensor, // [B, nv, dk, dv]
    chunk_size: usize,
) -> Result<Option<Tensor>> {
    let (batch, heads, seq_len, _) = q.dims4()?;
    let dtype = q.dtype();
    if chunk_size != 64
        || seq_len <= 1
        || seq_len % chunk_size != 0
        || dtype != DType::BF16
        || state.dtype() != DType::BF16
        || any_tensor_tracks_op(&[q, k, v, beta, g, state])
        || !backend.supports_gdn_full_chunk_forward_head_last()
    {
        return Ok(None);
    }

    let dv = v.dim(3)?;
    // Full chunks cover the whole sequence and the Metal kernel writes every
    // head-last output element exactly once.
    let out = unsafe { Tensor::empty((batch, seq_len, heads, dv), DType::BF16, q.device())? };

    for ci in 0..(seq_len / chunk_size) {
        let t_start = ci * chunk_size;
        let q_c = q.narrow(2, t_start, chunk_size)?.contiguous()?;
        let k_c = k.narrow(2, t_start, chunk_size)?.contiguous()?;
        let v_c = v.narrow(2, t_start, chunk_size)?.contiguous()?;
        let beta_c = beta.narrow(2, t_start, chunk_size)?.contiguous()?;
        let g_c = g.narrow(2, t_start, chunk_size)?.contiguous()?;

        let k_t_mat = k_c.transpose(2, 3)?.contiguous()?; // [B, nv, dk, C]
        let ks_entry = k_c.matmul(&*state)?; // [B, nv, C, dv]
        let kkt = k_c.matmul(&k_t_mat)?; // [B, nv, C, C]
        let qkt = q_c.matmul(&k_t_mat)?; // [B, nv, C, C]
        let q_s = q_c.matmul(&*state)?; // [B, nv, C, dv]

        if !backend.gdn_full_chunk_forward_head_last_into(
            &g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s, &beta_c, &k_t_mat, state, &out, t_start,
            seq_len,
        )? {
            if ci == 0 {
                return Ok(None);
            }
            anyhow::bail!("backend declined GDN head-last full-chunk path mid-sequence");
        }
    }

    Ok(Some(out))
}

/// Gated DeltaNet (GDN) linear attention forward pass.
///
/// Implements the recurrent linear attention mechanism used by 24/32 layers in Qwen3.5-4B.
/// Uses data-dependent gating (alpha/beta) and a delta rule update for the recurrent state.
///
/// `x`: [batch, seq_len, hidden_size]
/// `weights`: linear attention projection weights
/// `config`: model configuration
/// `recurrent_state`: [batch, nv, dk, dv] — mutable recurrent state, updated in place
/// `conv_state`: [batch, conv_dim, kernel_size-1] — mutable conv buffer, updated in place
///
/// Returns: [batch, seq_len, hidden_size]

/// Candle-op reference path for the Step-6 GDN gates. This is the original
/// Phase-6 implementation; it's kept as a fallback for shapes/dtypes outside
/// the fused kernel's envelope and as the algorithmic oracle for parity tests.
///
/// beta = sigmoid(b)                                // bf16
/// g    = -exp(A_log) * softplus(a + dt_bias)       // bf16 (F32 intermediates)
fn gated_deltanet_gates_fallback(
    a: &Tensor,
    b: &Tensor,
    weights: &GpuLinearAttentionWeights,
    input_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let beta = cuda_sigmoid(b).context("gdn gates fallback beta cuda_sigmoid")?; // [B, T, nv], bf16
    let a_f32 = a
        .to_dtype(DType::F32)
        .context("gdn gates fallback a to f32")?;
    let a_log_f32 = weights
        .a_log
        .to_dtype(DType::F32)
        .context("gdn gates fallback a_log to f32")?;
    let dt_bias_f32 = weights
        .dt_bias
        .to_dtype(DType::F32)
        .context("gdn gates fallback dt_bias to f32")?;
    let g = {
        let a_biased = a_f32
            .broadcast_add(&dt_bias_f32)
            .context("gdn gates fallback broadcast_add dt_bias")?;
        let sp = softplus(&a_biased).context("gdn gates fallback softplus")?;
        let neg_decay = a_log_f32
            .exp()
            .context("gdn gates fallback a_log exp")?
            .neg()
            .context("gdn gates fallback a_log neg")?; // -exp(A_log)
        sp.broadcast_mul(&neg_decay)
            .context("gdn gates fallback broadcast_mul neg_decay")?
    }
    .to_dtype(input_dtype)
    .context("gdn gates fallback output to input dtype")?;
    Ok((beta, g))
}

pub fn gated_deltanet_forward(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    weights: &GpuLinearAttentionWeights,
    config: &kiln_core::config::ModelConfig,
    recurrent_state: &mut Tensor,
    conv_state: &mut Tensor,
    capture_b11_taps: bool,
    capture_c41_taps: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    gated_deltanet_forward_decode_if(
        backend,
        x,
        weights,
        config,
        recurrent_state,
        conv_state,
        capture_b11_taps,
        capture_c41_taps,
        true,
        false,
        None,
        true,
        true,
        lora,
    )
}

/// GDN attention subblock through its residual add, excluding the following
/// MLP. Used by exact training-time split backprop to keep the recurrent GDN
/// graph separate from the MLP graph while preserving full-context state.
pub fn gdn_attention_residual_block(
    backend: &dyn BackendRuntime,
    hidden: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    recurrent_state: &mut Tensor,
    conv_state: &mut Tensor,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let GpuAttentionWeights::Linear(lin_weights) = &layer.attention else {
        anyhow::bail!("gdn_attention_residual_block called on a non-GDN layer");
    };
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_attn");
        rms_norm(hidden, &layer.input_layernorm, config.rms_norm_eps)?
    };
    let attn_out = gated_deltanet_forward(
        backend,
        &normed,
        lin_weights,
        config,
        recurrent_state,
        conv_state,
        false,
        false,
        lora,
    )?;
    (hidden + &attn_out).map_err(Into::into)
}

/// Pre-attention RMSNorm for a GDN layer.
pub fn gdn_attention_input_norm(
    hidden: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    rms_norm(hidden, &layer.input_layernorm, config.rms_norm_eps)
}

/// Input-projection outputs for the GDN attention subblock.
pub struct GdnInputProjectionParts {
    pub mixed_qkv: Tensor,
    pub z: Tensor,
    pub a: Tensor,
    pub b: Tensor,
}

pub fn gdn_attention_in_projections(
    backend: &dyn BackendRuntime,
    normed: &Tensor,
    weights: &GpuLinearAttentionWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<GdnInputProjectionParts> {
    let (lora_layer, lora_scale) = match lora {
        Some((layer, scale)) => (Some(layer), scale),
        None => (None, 0.0),
    };
    let mixed_qkv = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        normed,
        &weights.in_proj_qkv_t,
        lora_layer.and_then(|layer| layer.in_proj_qkv.as_ref()),
        lora_scale,
    )?;
    let z = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        normed,
        &weights.in_proj_z_t,
        lora_layer.and_then(|layer| layer.in_proj_z.as_ref()),
        lora_scale,
    )?;
    let a = gdn_in_proj_matmul(backend, normed, &weights.in_proj_a_t)?;
    let b = gdn_in_proj_matmul(backend, normed, &weights.in_proj_b_t)?;
    Ok(GdnInputProjectionParts { mixed_qkv, z, a, b })
}

/// Q/K/V tensors in `[B, nv, T, *]` layout after causal conv, GQA expansion,
/// and Q/K L2 normalization.
pub struct GdnQkvParts {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

pub fn gdn_qkv_from_mixed_training(
    _backend: &dyn BackendRuntime,
    mixed_qkv: &Tensor,
    weights: &GpuLinearAttentionWeights,
    config: &kiln_core::config::ModelConfig,
    conv_state: &mut Tensor,
) -> Result<GdnQkvParts> {
    let (batch, seq_len, _) = mixed_qkv.dims3()?;
    let input_dtype = mixed_qkv.dtype();
    let nk = config.linear_num_key_heads;
    let dk = config.linear_key_head_dim;
    let nv = config.linear_num_value_heads;
    let dv = config.linear_value_head_dim;
    let qk_dim = config.linear_qk_dim();
    let v_dim = config.linear_v_dim();
    let kernel_size = config.linear_conv_kernel_dim;
    let gqa_ratio = nv / nk;
    let scale = 1.0 / (dk as f64).sqrt();

    let mixed_qkv_ct = mixed_qkv.transpose(1, 2)?.contiguous()?;
    let post_silu = if seq_len > 1 {
        let y = causal_conv1d_prefill(&mixed_qkv_ct, &weights.conv1d, conv_state, kernel_size)?;
        cuda_silu(&y)?
    } else {
        let y = causal_conv1d_decode(&mixed_qkv_ct, &weights.conv1d, conv_state, kernel_size)?;
        cuda_silu(&y.to_dtype(DType::F32)?)?
    };
    let mixed_qkv = post_silu.transpose(1, 2)?;
    let q = mixed_qkv
        .narrow(2, 0, qk_dim)?
        .reshape((batch, seq_len, nk, dk))?;
    let k = mixed_qkv
        .narrow(2, qk_dim, qk_dim)?
        .reshape((batch, seq_len, nk, dk))?;
    let v = mixed_qkv
        .narrow(2, 2 * qk_dim, v_dim)?
        .reshape((batch, seq_len, nv, dv))?
        .to_dtype(input_dtype)?;
    let (q, k) = if gqa_ratio > 1 {
        let q = q
            .unsqueeze(3)?
            .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
            .contiguous()?
            .reshape((batch, seq_len, nv, dk))?;
        let k = k
            .unsqueeze(3)?
            .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
            .contiguous()?
            .reshape((batch, seq_len, nv, dk))?;
        (q, k)
    } else {
        (q.contiguous()?, k.contiguous()?)
    };
    let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
    Ok(GdnQkvParts {
        q: q.transpose(1, 2)?,
        k: k.transpose(1, 2)?,
        v: v.transpose(1, 2)?,
    })
}

pub fn gdn_gates_from_ab_training(
    a: &Tensor,
    b: &Tensor,
    weights: &GpuLinearAttentionWeights,
    input_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let (beta, g) = gated_deltanet_gates_fallback(a, b, weights, input_dtype)?;
    Ok((beta.transpose(1, 2)?, g.transpose(1, 2)?))
}

pub fn gdn_recurrent_forward_from_parts(
    backend: &dyn BackendRuntime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    recurrent_state: &mut Tensor,
) -> Result<Tensor> {
    let input_dtype = q.dtype();
    let state_external_dtype = recurrent_state.dtype();
    if state_external_dtype != input_dtype {
        *recurrent_state = recurrent_state.to_dtype(input_dtype)?;
    }

    let (out, head_last) = if let Some(attn_out) =
        gdn_recurrent_prefill_head_last(backend, q, k, v, beta, g, recurrent_state)?
    {
        (attn_out, true)
    } else {
        match gdn_chunkwise_recurrence_head_last_full_chunks(
            backend,
            q,
            k,
            v,
            beta,
            g,
            recurrent_state,
            GDN_CHUNK_SIZE,
        )? {
            Some(attn_out) => (attn_out, true),
            None => (
                gdn_chunkwise_recurrence(
                    backend,
                    q,
                    k,
                    v,
                    beta,
                    g,
                    recurrent_state,
                    GDN_CHUNK_SIZE,
                )?,
                false,
            ),
        }
    };

    if state_external_dtype != input_dtype {
        *recurrent_state = recurrent_state.to_dtype(state_external_dtype)?;
    }

    if head_last {
        Ok(out.transpose(1, 2)?)
    } else {
        Ok(out)
    }
}

pub fn gdn_gated_norm_from_recurrent(
    backend: &dyn BackendRuntime,
    recurrent_out_head_major: &Tensor,
    z: &Tensor,
    weights: &GpuLinearAttentionWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    let (batch, heads, seq_len, dv) = recurrent_out_head_major.dims4()?;
    let attn_out = recurrent_out_head_major.transpose(1, 2)?;
    let z = z.reshape((batch, seq_len, heads, dv))?;
    Ok(
        gated_rms_norm(backend, &attn_out, &z, &weights.norm, config.rms_norm_eps)?
            .reshape((batch, seq_len, heads * dv))?
            .to_dtype(z.dtype())?,
    )
}

pub fn gdn_out_proj_from_gated_norm(
    backend: &dyn BackendRuntime,
    normed: &Tensor,
    weights: &GpuLinearAttentionWeights,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (lora_layer, lora_scale) = match lora {
        Some((layer, scale)) => (Some(layer), scale),
        None => (None, 0.0),
    };
    mlp_proj_forward_decode_if(
        Some(backend),
        false,
        normed,
        &weights.out_proj_t,
        weights.out_proj_marlin.as_ref(),
        lora_layer.and_then(|layer| layer.gdn_out_proj.as_ref()),
        lora_scale,
    )
}

pub struct GdnRecurrentBackwardGrads {
    pub dq: Tensor,
    pub dk: Tensor,
    pub dv: Tensor,
    pub dbeta: Tensor,
    pub dg: Tensor,
    pub d_state: Option<Tensor>,
}

#[allow(clippy::too_many_arguments)]
fn gdn_chunk_prep_f32(
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let (batch, heads, chunk, _) = v.dims4()?;
    let device = v.device();
    let g_f32 = g.to_dtype(DType::F32)?;
    let big_g = g_f32.cumsum(candle_core::D::Minus1)?;
    let big_g_col = big_g.unsqueeze(3)?;
    let big_g_row = big_g.unsqueeze(2)?;
    let decay_delta = big_g_col.broadcast_sub(&big_g_row)?;
    let zero_delta = Tensor::zeros_like(&decay_delta)?;
    let strict_bool = strict_lower_tri_bool(chunk, device)?
        .reshape((1, 1, chunk, chunk))?
        .broadcast_as((batch, heads, chunk, chunk))?;
    let causal_bool = causal_lower_tri_bool(chunk, device)?
        .reshape((1, 1, chunk, chunk))?
        .broadcast_as((batch, heads, chunk, chunk))?;
    let strict_decay = strict_bool.where_cond(&decay_delta, &zero_delta)?.exp()?;
    let causal_decay = causal_bool.where_cond(&decay_delta, &zero_delta)?.exp()?;
    let p = big_g.exp()?;
    let p_col = p.unsqueeze(3)?;
    let strict_mask = strict_bool.to_dtype(DType::F32)?;
    let causal_mask = causal_bool.to_dtype(DType::F32)?;

    let v_f32 = v.to_dtype(DType::F32)?;
    let kkt_f32 = kkt.to_dtype(DType::F32)?;
    let qkt_f32 = qkt.to_dtype(DType::F32)?;
    let ks_entry_f32 = ks_entry.to_dtype(DType::F32)?;
    let q_s_f32 = q_s.to_dtype(DType::F32)?;
    let v_prime = (&v_f32 - ks_entry_f32.broadcast_mul(&p_col)?)?;
    let a_strict = kkt_f32
        .broadcast_mul(&strict_decay)?
        .broadcast_mul(&strict_mask)?
        .contiguous()?;
    let b_mask = qkt_f32
        .broadcast_mul(&causal_decay)?
        .broadcast_mul(&causal_mask)?
        .contiguous()?;
    let q_s_scaled = q_s_f32.broadcast_mul(&p_col)?;
    let g_last = big_g.narrow(2, chunk - 1, 1)?;
    let decay_last_col = g_last.broadcast_sub(&big_g)?.exp()?;
    let p_last = g_last.exp()?.squeeze(2)?;
    Ok((
        a_strict,
        b_mask,
        v_prime,
        q_s_scaled,
        decay_last_col,
        p_last,
    ))
}

fn solve_tri_transpose_f32(a_strict: &Tensor, beta: &Tensor, dw: &Tensor) -> Result<Tensor> {
    let (_, _, chunk, _) = dw.dims4()?;
    let mut rows_rev: Vec<Tensor> = Vec::with_capacity(chunk);
    for t in (0..chunk).rev() {
        let dw_t = dw.narrow(2, t, 1)?;
        let dr_t = if rows_rev.is_empty() {
            dw_t
        } else {
            let future_len = chunk - t - 1;
            let mut future_refs: Vec<&Tensor> = Vec::with_capacity(future_len);
            for row in rows_rev.iter().rev() {
                future_refs.push(row);
            }
            let dr_future = Tensor::cat(&future_refs, 2)?;
            let a_col = a_strict.narrow(2, t + 1, future_len)?.narrow(3, t, 1)?;
            let beta_future = beta.narrow(2, t + 1, future_len)?.unsqueeze(3)?;
            let weights = a_col.broadcast_mul(&beta_future)?;
            let acc = dr_future.broadcast_mul(&weights)?.sum(2)?.unsqueeze(2)?;
            (dw_t - acc)?
        };
        rows_rev.push(dr_t);
    }
    rows_rev.reverse();
    let refs: Vec<&Tensor> = rows_rev.iter().collect();
    Ok(Tensor::cat(&refs, 2)?)
}

fn reverse_cumsum_time(x: &Tensor) -> Result<Tensor> {
    let chunk = x.dim(2)?;
    let mut rows_rev: Vec<Tensor> = Vec::with_capacity(chunk);
    let mut acc: Option<Tensor> = None;
    for t in (0..chunk).rev() {
        let x_t = x.narrow(2, t, 1)?;
        let next = match acc {
            Some(prev) => (&prev + &x_t)?,
            None => x_t,
        };
        rows_rev.push(next.clone());
        acc = Some(next);
    }
    rows_rev.reverse();
    let refs: Vec<&Tensor> = rows_rev.iter().collect();
    Ok(Tensor::cat(&refs, 2)?)
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_backward_no_grad(
    backend: &dyn BackendRuntime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    entry_state: &Tensor,
    grad_out: &Tensor,
    grad_exit_state: Option<&Tensor>,
    chunk_size: usize,
) -> Result<GdnRecurrentBackwardGrads> {
    let (batch, heads, seq_len, _dk) = q.dims4()?;
    let full_chunks = seq_len / chunk_size;
    let tail = seq_len - full_chunks * chunk_size;
    let total_chunks = full_chunks + if tail > 0 { 1 } else { 0 };

    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let beta = beta.to_dtype(DType::F32)?;
    let g = g.to_dtype(DType::F32)?;
    let grad_out = grad_out.to_dtype(DType::F32)?;
    let mut state = entry_state.to_dtype(DType::F32)?;
    let mut state_snapshots: Vec<Tensor> = Vec::with_capacity(total_chunks);

    for ci in 0..total_chunks {
        let chunk = if ci >= full_chunks { tail } else { chunk_size };
        let t_off = ci * chunk_size;
        state_snapshots.push(state.clone());
        let q_c = q.narrow(2, t_off, chunk)?.contiguous()?;
        let k_c = k.narrow(2, t_off, chunk)?.contiguous()?;
        let v_c = v.narrow(2, t_off, chunk)?.contiguous()?;
        let beta_c = beta.narrow(2, t_off, chunk)?.contiguous()?;
        let g_c = g.narrow(2, t_off, chunk)?.contiguous()?;
        let k_t = k_c.transpose(2, 3)?.contiguous()?;
        let ks_entry = k_c.matmul(&state)?;
        let q_s = q_c.matmul(&state)?;
        let kkt = k_c.matmul(&k_t)?;
        let qkt = q_c.matmul(&k_t)?;
        let (a_strict, _b_mask, v_prime, _q_s_scaled, decay_last_col, p_last) =
            gdn_chunk_prep_f32(&g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s)?;
        let w = compute_w_chunk(backend, &a_strict, &v_prime, &beta_c, chunk)?.contiguous()?;
        let state_scaled = state.broadcast_mul(&p_last.unsqueeze(2)?.unsqueeze(3)?)?;
        let w_weighted = w.broadcast_mul(&decay_last_col.unsqueeze(3)?)?;
        let delta_state = k_t.matmul(&w_weighted)?;
        state = (state_scaled + delta_state)?;
    }

    let mut dq_chunks: Vec<Option<Tensor>> = (0..total_chunks).map(|_| None).collect();
    let mut dk_chunks: Vec<Option<Tensor>> = (0..total_chunks).map(|_| None).collect();
    let mut dv_chunks: Vec<Option<Tensor>> = (0..total_chunks).map(|_| None).collect();
    let mut dbeta_chunks: Vec<Option<Tensor>> = (0..total_chunks).map(|_| None).collect();
    let mut dg_chunks: Vec<Option<Tensor>> = (0..total_chunks).map(|_| None).collect();
    let mut d_s_carry = match grad_exit_state {
        Some(grad) => Some(grad.to_dtype(DType::F32)?),
        None => None,
    };

    for ci in (0..total_chunks).rev() {
        let chunk = if ci >= full_chunks { tail } else { chunk_size };
        let t_off = ci * chunk_size;
        let s_in = &state_snapshots[ci];
        let q_c = q.narrow(2, t_off, chunk)?.contiguous()?;
        let k_c = k.narrow(2, t_off, chunk)?.contiguous()?;
        let v_c = v.narrow(2, t_off, chunk)?.contiguous()?;
        let beta_c = beta.narrow(2, t_off, chunk)?.contiguous()?;
        let g_c = g.narrow(2, t_off, chunk)?.contiguous()?;
        let d_out = grad_out.narrow(2, t_off, chunk)?.contiguous()?;
        let k_t = k_c.transpose(2, 3)?.contiguous()?;
        let ks_entry = k_c.matmul(s_in)?;
        let q_s = q_c.matmul(s_in)?;
        let kkt = k_c.matmul(&k_t)?;
        let qkt = q_c.matmul(&k_t)?;
        let (a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last) =
            gdn_chunk_prep_f32(&g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s)?;
        let w = compute_w_chunk(backend, &a_strict, &v_prime, &beta_c, chunk)?.contiguous()?;

        let dq_s_scaled = d_out.clone();
        let d_w_scan = b_mask.transpose(2, 3)?.contiguous()?.matmul(&d_out)?;
        let d_b_mask = d_out.matmul(&w.transpose(2, 3)?.contiguous()?)?;

        let mut d_w_acc = d_w_scan;
        let mut d_decay_last_col_acc =
            Tensor::zeros((batch, heads, chunk), DType::F32, q.device())?;
        let mut d_p_last_acc = Tensor::zeros((batch, heads), DType::F32, q.device())?;
        let mut dk_state_extra: Option<Tensor> = None;
        let mut ds_state_extra: Option<Tensor> = None;

        if let Some(d_s_exit) = d_s_carry.as_ref() {
            let p_last_u = p_last.unsqueeze(2)?.unsqueeze(3)?;
            ds_state_extra = Some(d_s_exit.broadcast_mul(&p_last_u)?);
            d_p_last_acc = (s_in * d_s_exit)?.sum(3)?.sum(2)?;
            let tmp_dw = k_c.matmul(d_s_exit)?;
            d_w_acc = (&d_w_acc + &tmp_dw.broadcast_mul(&decay_last_col.unsqueeze(3)?)?)?;
            let tmp_dk = w.matmul(&d_s_exit.transpose(2, 3)?.contiguous()?)?;
            dk_state_extra = Some(tmp_dk.broadcast_mul(&decay_last_col.unsqueeze(3)?)?);
            d_decay_last_col_acc = (&k_c * &tmp_dk)?.sum(candle_core::D::Minus1)?;
        }

        let dr = solve_tri_transpose_f32(&a_strict, &beta_c, &d_w_acc)?.contiguous()?;
        let a_w = a_strict.matmul(&w)?;
        let pre_beta = (&v_prime - &a_w)?;
        let d_v_prime = dr.broadcast_mul(&beta_c.unsqueeze(3)?)?.contiguous()?;
        let d_beta = (&pre_beta * &dr)?.sum(candle_core::D::Minus1)?;
        let dr_w_t = dr.matmul(&w.transpose(2, 3)?.contiguous()?)?;
        let strict_mask = strict_lower_tri_bool(chunk, q.device())?
            .reshape((1, 1, chunk, chunk))?
            .broadcast_as((batch, heads, chunk, chunk))?
            .to_dtype(DType::F32)?;
        let d_a_strict = dr_w_t
            .broadcast_mul(&beta_c.neg()?.unsqueeze(3)?)?
            .broadcast_mul(&strict_mask)?;

        let big_g = g_c.cumsum(candle_core::D::Minus1)?;
        let p = big_g.exp()?;
        let p_col = p.unsqueeze(3)?;
        let d_v = d_v_prime.clone();
        let d_ks_entry = d_v_prime.broadcast_mul(&p_col)?.neg()?.contiguous()?;
        let mut d_g_acc = (&ks_entry * &d_ks_entry)?.sum(candle_core::D::Minus1)?;
        let d_q_s = dq_s_scaled.broadcast_mul(&p_col)?.contiguous()?;
        d_g_acc = (&d_g_acc
            + &(&q_s * &dq_s_scaled)?
                .broadcast_mul(&p_col)?
                .sum(candle_core::D::Minus1)?)?;

        let big_g_col = big_g.unsqueeze(3)?;
        let big_g_row = big_g.unsqueeze(2)?;
        let decay_delta = big_g_col.broadcast_sub(&big_g_row)?;
        let zero_delta = Tensor::zeros_like(&decay_delta)?;
        let strict_bool = strict_lower_tri_bool(chunk, q.device())?
            .reshape((1, 1, chunk, chunk))?
            .broadcast_as((batch, heads, chunk, chunk))?;
        let causal_bool = causal_lower_tri_bool(chunk, q.device())?
            .reshape((1, 1, chunk, chunk))?
            .broadcast_as((batch, heads, chunk, chunk))?;
        let strict_decay = strict_bool
            .where_cond(&decay_delta, &zero_delta)?
            .exp()?
            .broadcast_mul(&strict_bool.to_dtype(DType::F32)?)?;
        let causal_decay = causal_bool
            .where_cond(&decay_delta, &zero_delta)?
            .exp()?
            .broadcast_mul(&causal_bool.to_dtype(DType::F32)?)?;
        let d_kkt = d_a_strict.broadcast_mul(&strict_decay)?.contiguous()?;
        let d_qkt = d_b_mask.broadcast_mul(&causal_decay)?.contiguous()?;
        let term_a = d_a_strict
            .broadcast_mul(&strict_decay)?
            .broadcast_mul(&kkt)?;
        let term_b = d_b_mask.broadcast_mul(&causal_decay)?.broadcast_mul(&qkt)?;
        let term = (&term_a + &term_b)?;
        let row_sum = term.sum(candle_core::D::Minus1)?;
        let col_sum = term.sum(2)?;
        d_g_acc = (&d_g_acc + &row_sum)?;
        d_g_acc = (&d_g_acc - &col_sum)?;

        let decay_term = decay_last_col.broadcast_mul(&d_decay_last_col_acc)?;
        let decay_sum = decay_term.sum(candle_core::D::Minus1)?.unsqueeze(2)?;
        let last_mask = Tensor::arange(0u32, chunk as u32, q.device())?
            .eq((chunk - 1) as u32)?
            .to_dtype(DType::F32)?
            .reshape((1, 1, chunk))?
            .broadcast_as((batch, heads, chunk))?;
        d_g_acc = (&d_g_acc - &decay_term)?;
        d_g_acc = (&d_g_acc + &decay_sum.broadcast_mul(&last_mask)?)?;
        let p_last_term = p
            .narrow(2, chunk - 1, 1)?
            .squeeze(2)?
            .broadcast_mul(&d_p_last_acc)?
            .unsqueeze(2)?
            .broadcast_mul(&last_mask)?;
        d_g_acc = (&d_g_acc + &p_last_term)?;
        let d_g = reverse_cumsum_time(&d_g_acc)?;

        let s_t = s_in.transpose(2, 3)?.contiguous()?;
        let d_k_from_kkt =
            (&d_kkt.matmul(&k_c)? + &d_kkt.transpose(2, 3)?.contiguous()?.matmul(&k_c)?)?;
        let d_k_from_qkt = d_qkt.transpose(2, 3)?.contiguous()?.matmul(&q_c)?;
        let d_k_from_ks = d_ks_entry.matmul(&s_t)?;
        let mut d_k = (&(&d_k_from_kkt + &d_k_from_qkt)? + &d_k_from_ks)?;
        if let Some(extra) = dk_state_extra.as_ref() {
            d_k = (&d_k + extra)?;
        }
        let d_q = (&d_qkt.matmul(&k_c)? + &d_q_s.matmul(&s_t)?)?;
        let d_s_from_ks = k_c.transpose(2, 3)?.contiguous()?.matmul(&d_ks_entry)?;
        let d_s_from_qs = q_c.transpose(2, 3)?.contiguous()?.matmul(&d_q_s)?;
        let mut d_s_in = (&d_s_from_ks + &d_s_from_qs)?;
        if let Some(extra) = ds_state_extra.as_ref() {
            d_s_in = (&d_s_in + extra)?;
        }

        dq_chunks[ci] = Some(d_q);
        dk_chunks[ci] = Some(d_k);
        dv_chunks[ci] = Some(d_v);
        dbeta_chunks[ci] = Some(d_beta);
        dg_chunks[ci] = Some(d_g);
        d_s_carry = Some(d_s_in);

        let _ = q_s_scaled;
    }

    let collect = |chunks: &[Option<Tensor>], name: &str| -> Result<Tensor> {
        let mut refs = Vec::with_capacity(chunks.len());
        for (idx, chunk) in chunks.iter().enumerate() {
            refs.push(
                chunk
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing {name} chunk {idx}"))?,
            );
        }
        Ok(Tensor::cat(&refs, 2)?)
    };

    Ok(GdnRecurrentBackwardGrads {
        dq: collect(&dq_chunks, "dq")?,
        dk: collect(&dk_chunks, "dk")?,
        dv: collect(&dv_chunks, "dv")?,
        dbeta: collect(&dbeta_chunks, "dbeta")?,
        dg: collect(&dg_chunks, "dg")?,
        d_state: d_s_carry,
    })
}

/// Streaming/tiled wrapper around [`gated_deltanet_forward`] for the
/// training-time forward path.
///
/// Slices `x: [B, T, hidden]` along T into tiles of `tile_size` (the last
/// tile may be partial), calls [`gated_deltanet_forward`] per tile threading
/// `recurrent_state` and `conv_state` across tile boundaries, and
/// concatenates the per-tile outputs back into `[B, T, hidden]` along T.
///
/// Tiling reduces peak transient activation memory: GDN's F32 intermediates
/// inside the conv1d / l2_normalize / chunkwise paths allocate per-call
/// buffers sized by the input length, so smaller tiles → smaller transient
/// allocations. The `LinearAttentionState` recurrent + conv state hand-off
/// makes this bit-exact with the monolithic call by construction
/// (the inference path uses the same hand-off in
/// [`model_forward_paged_streaming_with`]).
///
/// `tile_size` must be a positive multiple of `GDN_CHUNK_SIZE`. The last
/// tile may be smaller; partial tile lengths are handled by
/// [`gated_deltanet_forward`] itself (the same way the inference streaming
/// path handles a non-aligned final tile).
///
/// Used by [`model_forward_segment`] when `KILN_STREAMING_PREFILL=1` is set
/// and the segment's seq_len exceeds `tile_size`.
pub fn gated_deltanet_forward_streaming(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    weights: &GpuLinearAttentionWeights,
    config: &kiln_core::config::ModelConfig,
    recurrent_state: &mut Tensor,
    conv_state: &mut Tensor,
    tile_size: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    if tile_size == 0 || tile_size % GDN_CHUNK_SIZE != 0 {
        anyhow::bail!(
            "streaming tile_size must be a positive multiple of GDN_CHUNK_SIZE ({}), got {tile_size}",
            GDN_CHUNK_SIZE
        );
    }
    let (_b, total, _h) = x.dims3()?;
    if total == 0 {
        anyhow::bail!("gated_deltanet_forward_streaming requires at least one token");
    }
    if total <= tile_size {
        // Single tile — no benefit from the cat overhead, defer to the
        // monolithic path so behavior matches the env-off case bit-exactly.
        return gated_deltanet_forward(
            backend,
            x,
            weights,
            config,
            recurrent_state,
            conv_state,
            false,
            false,
            lora,
        );
    }

    let cap = total.div_ceil(tile_size);
    let mut tile_outs: Vec<Tensor> = Vec::with_capacity(cap);
    let tile_device = x.device();
    let mut cursor = 0usize;
    while cursor < total {
        let end = (cursor + tile_size).min(total);
        let len = end - cursor;
        let allow_forward_only_fastpaths =
            streaming_gdn_forward_only_fastpaths_allowed(tile_device);
        let allow_prefill_recurrent_kernel = allow_forward_only_fastpaths;
        let mut run_tile = || -> Result<Tensor> {
            let tile_in = x.narrow(1, cursor, len)?;
            gated_deltanet_forward_decode_if(
                backend,
                &tile_in,
                weights,
                config,
                recurrent_state,
                conv_state,
                false,
                false,
                true,
                false,
                None,
                allow_forward_only_fastpaths,
                allow_prefill_recurrent_kernel,
                lora,
            )
            .with_context(|| {
                format!("streaming GDN tile [{cursor}, {end}) of {total} (tile_size={tile_size})")
            })
        };
        let tile_out = {
            #[cfg(feature = "metal")]
            {
                if matches!(tile_device, Device::Metal(_)) {
                    let tile_out = metal_autoreleasepool(|| run_tile())?;
                    tile_device.synchronize().with_context(|| {
                        format!("synchronize streaming GDN tile [{cursor}, {end}) of {total}")
                    })?;
                    tile_out
                } else {
                    run_tile()?
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                run_tile()?
            }
        };
        tile_outs.push(tile_out);
        cursor = end;
    }

    let tile_refs: Vec<&Tensor> = tile_outs.iter().collect();
    Tensor::cat(&tile_refs, 1).context("streaming GDN cat tile outputs along T axis")
}

#[allow(clippy::too_many_arguments)]
fn gated_deltanet_forward_decode_if(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    weights: &GpuLinearAttentionWeights,
    config: &kiln_core::config::ModelConfig,
    recurrent_state: &mut Tensor,
    conv_state: &mut Tensor,
    capture_b11_taps: bool,
    capture_c41_taps: bool,
    use_fused_gdn_gates: bool,
    use_metal_decode_gemv: bool,
    profile_context: Option<(usize, usize)>,
    allow_forward_only_fastpaths: bool,
    allow_prefill_recurrent_kernel: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (batch, seq_len, _hidden) = x.dims3()?;
    let profile_device = x.device();
    let input_dtype = x.dtype();
    let nk = config.linear_num_key_heads;
    let dk = config.linear_key_head_dim;
    let nv = config.linear_num_value_heads;
    let dv = config.linear_value_head_dim;
    let qk_dim = config.linear_qk_dim();
    let v_dim = config.linear_v_dim();
    let kernel_size = config.linear_conv_kernel_dim;
    let gqa_ratio = nv / nk;
    let gdn_forward_only_fastpaths = allow_forward_only_fastpaths && !x.track_op();
    let (lora_layer, lora_scale) = match lora {
        Some((layer, scale)) => (Some(layer), scale),
        None => (None, 0.0),
    };
    let in_proj_qkv_lora = lora_layer.and_then(|l| l.in_proj_qkv.as_ref());
    let in_proj_z_lora = lora_layer.and_then(|l| l.in_proj_z.as_ref());
    let gdn_out_lora = lora_layer.and_then(|l| l.gdn_out_proj.as_ref());
    let has_gdn_in_lora = in_proj_qkv_lora.is_some() || in_proj_z_lora.is_some();
    // --- Step 1: Input projections ---
    // Use the pre-transposed weight cache (Phase 6) so we don't pay a `.t().contiguous()`
    // ucopy_bf16 copy on every layer / every step. Same fix class as PR #128 (MLP/full-attn).
    let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
    let (mixed_qkv, z, a, b, prefill_ab_for_gates) = {
        kiln_nvtx::range!(c"kiln/gdn/in_proj");
        if !has_gdn_in_lora
            && gdn_forward_only_fastpaths
            && let Some((mixed_qkv, z, a, b)) = backend.gdn_in_proj_decode(
                x,
                &weights.in_proj_qkv_t,
                &weights.in_proj_z_t,
                &weights.in_proj_a_t,
                &weights.in_proj_b_t,
            )?
        {
            (mixed_qkv, z, a, b, None::<Tensor>)
        } else {
            let mixed_qkv = linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                x,
                &weights.in_proj_qkv_t,
                in_proj_qkv_lora,
                lora_scale,
            )?; // [B, T, qkv_dim]
            let z = linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                x,
                &weights.in_proj_z_t,
                in_proj_z_lora,
                lora_scale,
            )?; // [B, T, v_dim]
            let prefill_ab: Option<(Tensor, Tensor, Tensor)> = {
                #[cfg(any(feature = "cuda", feature = "metal"))]
                {
                    let mut out = None;
                    if let Some(in_proj_ab_t) = weights.in_proj_ab_t.as_ref() {
                        #[cfg(feature = "metal")]
                        {
                            if out.is_none()
                                && gdn_forward_only_fastpaths
                                && crate::backend::metal::metal_gdn_prefill_ab_in_proj_supports(
                                    x,
                                    in_proj_ab_t,
                                    nv,
                                )
                            {
                                let (ab, a, b) =
                                    crate::backend::metal::metal_gdn_prefill_ab_in_proj_bf16(
                                        x,
                                        in_proj_ab_t,
                                        nv,
                                    )
                                    .context("metal gdn prefill A/B in-proj")?;
                                out = Some((ab, a, b));
                            }
                        }
                        #[cfg(feature = "cuda")]
                        {
                            if out.is_none()
                                && cuda_gdn_ab_in_proj_enabled()
                                && (seq_len == 1
                                    || (seq_len <= CUDA_GDN_PREFILL_AB_IN_PROJ_MAX_TOKENS
                                        && cuda_gdn_prefill_ab_in_proj_enabled()))
                                && gdn_forward_only_fastpaths
                                && seq_len >= 1
                                && x.dtype() == DType::BF16
                                && in_proj_ab_t.dtype() == DType::BF16
                                && !in_proj_ab_t.track_op()
                                && matches!(x.device(), Device::Cuda(_))
                                && matches!(in_proj_ab_t.device(), Device::Cuda(_))
                                && in_proj_ab_t.is_contiguous()
                                && in_proj_ab_t.dims() == [x.dim(2)?, 2 * nv]
                            {
                                let ab = broadcast_matmul_cpu_compatible(x, in_proj_ab_t)
                                    .context("cuda gdn combined A/B in-proj matmul")?;
                                let a = ab.narrow(2, 0, nv)?;
                                let b = ab.narrow(2, nv, nv)?;
                                out = Some((ab, a, b));
                            }
                        }
                    }
                    out
                }
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                {
                    None
                }
            };
            if let Some((ab, a, b)) = prefill_ab {
                (mixed_qkv, z, a, b, Some(ab))
            } else {
                let a = gdn_in_proj_matmul(backend, x, &weights.in_proj_a_t)?; // [B, T, nv]
                let b = gdn_in_proj_matmul(backend, x, &weights.in_proj_b_t)?; // [B, T, nv]
                (mixed_qkv, z, a, b, None::<Tensor>)
            }
        }
    };
    finish_gdn_stage_profile(
        profile_device,
        profile_context,
        "in_proj",
        seq_len,
        stage_profile,
    )?;
    #[cfg(not(feature = "metal"))]
    let _ = &prefill_ab_for_gates;

    // Phase B11b tap: `gdn_in_proj`. Matches the HF reference layout
    // `concat([in_proj_qkvz(x), in_proj_ba(x)], dim=-1)` = [q, k, v, z, b, a]
    // along the last axis. Capture once here so subsequent post-split
    // transforms don't alter what we're attributing divergence to.
    if capture_b11_taps {
        let gdn_in_proj = Tensor::cat(&[&mixed_qkv, &z, &b, &a], candle_core::D::Minus1)?;
        crate::mtp_debug::capture_b11_layer0_tap("gdn_in_proj", &gdn_in_proj)?;
    }
    if capture_c41_taps {
        let gdn_in_proj = Tensor::cat(&[&mixed_qkv, &z, &b, &a], candle_core::D::Minus1)?;
        crate::mtp_debug::capture_c41_layer1_tap("gdn_in_proj", &gdn_in_proj)?;
    }

    let scale = 1.0 / (dk as f64).sqrt();
    let recurrent_unexpanded_qk = input_dtype == DType::BF16
        && gdn_forward_only_fastpaths
        && seq_len >= 1
        && seq_len <= GDN_RECURRENT_PREFILL_MAX_TOKENS
        && dk == 128
        && gqa_ratio > 1
        && !capture_b11_taps
        && !capture_c41_taps
        && backend.supports_gdn_recurrent_prefill_native_head_last();
    let fused_decode_unexpanded_qk = input_dtype == DType::BF16
        && gdn_forward_only_fastpaths
        && seq_len == 1
        && dk == 128
        && gqa_ratio > 1
        && !capture_b11_taps
        && !capture_c41_taps
        && backend.supports_gdn_decode_gates_recurrent_unexpanded_qk();
    #[cfg(feature = "metal")]
    let use_unexpanded_qk = recurrent_unexpanded_qk || fused_decode_unexpanded_qk;
    let fused_decode_qkv_conv_norm = {
        #[cfg(feature = "metal")]
        {
            if use_unexpanded_qk
                && gdn_forward_only_fastpaths
                && !capture_b11_taps
                && !capture_c41_taps
                && crate::backend::metal::metal_gdn_decode_qkv_conv_norm_supports(
                    &mixed_qkv,
                    &weights.conv1d,
                    conv_state,
                    kernel_size,
                    nk,
                    dk,
                    nv,
                    dv,
                )
            {
                kiln_nvtx::range!(c"kiln/gdn/qkv_conv_norm");
                let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
                let (q, k, v) = crate::backend::metal::metal_gdn_decode_qkv_conv_norm_bf16(
                    &mixed_qkv,
                    &weights.conv1d,
                    conv_state,
                    kernel_size,
                    nk,
                    dk,
                    nv,
                    dv,
                    scale as f32,
                    1e-6,
                )
                .context("metal gdn decode qkv conv/norm kernel failed")?;
                let z = z.reshape((batch, seq_len, nv, dv))?;
                finish_gdn_stage_profile(
                    profile_device,
                    profile_context,
                    "qkv_conv_norm",
                    seq_len,
                    stage_profile,
                )?;
                Some((q, k, v, z, false, false, false))
            } else {
                None
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            None
        }
    };

    let fused_prefill_qkv_conv_split = {
        #[cfg(feature = "metal")]
        {
            if fused_decode_qkv_conv_norm.is_none()
                && recurrent_unexpanded_qk
                && gdn_forward_only_fastpaths
                && seq_len > 1
                && !capture_b11_taps
                && !capture_c41_taps
                && crate::backend::metal::metal_gdn_prefill_qkv_conv_split_supports(
                    &mixed_qkv,
                    &weights.conv1d,
                    conv_state,
                    kernel_size,
                    nk,
                    dk,
                    nv,
                    dv,
                )
            {
                kiln_nvtx::range!(c"kiln/gdn/qkv_conv_split");
                let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
                let (q, k, v) =
                    crate::backend::metal::metal_gdn_prefill_qkv_conv_split_bf16_f32_k4(
                        &mixed_qkv,
                        &weights.conv1d,
                        conv_state,
                        kernel_size,
                        nk,
                        dk,
                        nv,
                        dv,
                    )
                    .context("metal gdn prefill qkv conv-split kernel failed")?;
                let (q, k) = {
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_unexpanded");
                    gdn_qk_norm(&q, &k, input_dtype, scale)?
                };
                let z = z.reshape((batch, seq_len, nv, dv))?;
                finish_gdn_stage_profile(
                    profile_device,
                    profile_context,
                    "qkv_conv_split_norm",
                    seq_len,
                    stage_profile,
                )?;
                Some((q, k, v, z, false, false, false))
            } else {
                None
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            None
        }
    };

    let (
        q,
        k,
        v,
        z,
        qk_expanded,
        qk_norm_deferred_to_recurrent,
        qk_norm_deferred_to_native_recurrent,
    ) = if let Some(fused) = fused_decode_qkv_conv_norm {
        fused
    } else if let Some(fused) = fused_prefill_qkv_conv_split {
        fused
    } else {
        // --- Step 2: Causal depthwise conv1d + SiLU on fused QKV ---
        //
        // Decode fast path: backend-side `causal_conv1d_update` collapses the
        // to_f32 / cat / sum / narrow / silu chain into one fused update per
        // (batch, channel). It returns F32 with SiLU already fused, so the
        // subsequent `cuda_silu(.to_dtype(F32))` step is skipped. Unsupported
        // backends, non-bf16, kernel_size != 4, and the `KILN_DISABLE_FUSED_CONV1D`
        // kill switch all route through the portable candle path below — which is the
        // parity oracle.
        let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
        let mixed_qkv = {
            kiln_nvtx::range!(c"kiln/gdn/conv");
            // Transpose to [B, channels, T] for conv. At seq_len == 1 the
            // [B, 1, C] -> [B, C, 1] axis swap is a no-data-move shape
            // reinterpretation: in row-major, element[b, 0, c] sits at the
            // same offset as element[b, c, 0]. `reshape` on a contiguous
            // input produces a view (no copy); the conv kernel's strict
            // [B, C, 1] dims check accepts the view. Saves the
            // transpose + `contiguous` copy that nsys flagged as ~3 ms /
            // bench-bs=16 in `kiln/gdn/conv/layout`.
            let mixed_qkv_ct = {
                kiln_nvtx::range!(c"kiln/gdn/conv/layout");
                if seq_len == 1 && mixed_qkv.is_contiguous() {
                    let (b, _t, c) = mixed_qkv.dims3()?;
                    mixed_qkv.reshape((b, c, 1))?
                } else {
                    mixed_qkv.transpose(1, 2)?.contiguous()?
                }
            };
            let post_silu = if seq_len == 1
                && gdn_forward_only_fastpaths
                && backend.supports_causal_conv1d_update()
            {
                let conv_update = {
                    kiln_nvtx::range!(c"kiln/gdn/conv/update");
                    backend.causal_conv1d_update(
                        &mixed_qkv_ct,
                        &weights.conv1d,
                        conv_state,
                        kernel_size,
                    )?
                };
                match conv_update {
                    Some(out) => out, // F32, SiLU fused into the kernel epilogue
                    None => {
                        kiln_nvtx::range!(c"kiln/gdn/conv/fallback_decode");
                        let y = causal_conv1d_decode(
                            &mixed_qkv_ct,
                            &weights.conv1d,
                            conv_state,
                            kernel_size,
                        )?;
                        cuda_silu(&y.to_dtype(DType::F32)?)?
                    }
                }
            } else if seq_len > 1 {
                if gdn_forward_only_fastpaths && backend.supports_causal_conv1d_prefill() {
                    let conv_prefill = {
                        kiln_nvtx::range!(c"kiln/gdn/conv/prefill_update");
                        backend.causal_conv1d_prefill(
                            &mixed_qkv_ct,
                            &weights.conv1d,
                            conv_state,
                            kernel_size,
                        )?
                    };
                    match conv_prefill {
                        Some(out) => out, // F32, SiLU fused into the kernel epilogue
                        None => {
                            kiln_nvtx::range!(c"kiln/gdn/conv/fallback_prefill");
                            let y = causal_conv1d_prefill(
                                &mixed_qkv_ct,
                                &weights.conv1d,
                                conv_state,
                                kernel_size,
                            )?;
                            cuda_silu(&y)?
                        }
                    }
                } else {
                    kiln_nvtx::range!(c"kiln/gdn/conv/fallback_prefill");
                    let y = causal_conv1d_prefill(
                        &mixed_qkv_ct,
                        &weights.conv1d,
                        conv_state,
                        kernel_size,
                    )?;
                    cuda_silu(&y)?
                }
            } else {
                kiln_nvtx::range!(c"kiln/gdn/conv/fallback_decode");
                let y =
                    causal_conv1d_decode(&mixed_qkv_ct, &weights.conv1d, conv_state, kernel_size)?;
                cuda_silu(&y.to_dtype(DType::F32)?)?
            };
            // Transpose back to [B, T, qkv_dim]
            post_silu.transpose(1, 2)?
        };
        finish_gdn_stage_profile(
            profile_device,
            profile_context,
            "conv",
            seq_len,
            stage_profile,
        )?;

        // Phase B11b tap: `gdn_conv`. Output of the causal depthwise conv1d +
        // SiLU, matching HF's `mixed_qkv` after `self.conv1d(...)[:T]` +
        // `F.silu(...)` (shape [B, T, qkv_dim]).
        if capture_b11_taps {
            crate::mtp_debug::capture_b11_layer0_tap("gdn_conv", &mixed_qkv)?;
        }
        if capture_c41_taps {
            crate::mtp_debug::capture_c41_layer1_tap("gdn_conv", &mixed_qkv)?;
        }

        // --- Step 3: Split into Q, K, V and reshape to heads ---
        let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
        let (q, k, v, z) = {
            kiln_nvtx::range!(c"kiln/gdn/qkv_split");
            let q = mixed_qkv
                .narrow(2, 0, qk_dim)?
                .reshape((batch, seq_len, nk, dk))?;
            let k = mixed_qkv
                .narrow(2, qk_dim, qk_dim)?
                .reshape((batch, seq_len, nk, dk))?;
            let v = mixed_qkv
                .narrow(2, 2 * qk_dim, v_dim)?
                .reshape((batch, seq_len, nv, dv))?;
            let z = z.reshape((batch, seq_len, nv, dv))?;
            (q, k, v, z)
        };
        finish_gdn_stage_profile(
            profile_device,
            profile_context,
            "qkv_split",
            seq_len,
            stage_profile,
        )?;

        // --- Step 4/5: GQA head repeat (nk → nv), L2 normalize Q/K, scale Q ---
        //
        // Fast paths: Metal and CUDA default to fused F32->BF16 kernels for
        // supported bf16 tensors. Both collapse the l2-normalize(Q) + scale(Q) +
        // l2-normalize(K) + dtype-cast chain (~11 candle launches on tiny per-row
        // tensors at decode shape) into a single launch. CUDA can be forced back
        // to the candle parity path with `KILN_DISABLE_FUSED_L2_QK_NORM=1`.
        //
        // Both paths produce bf16 outputs in `input_dtype`; only the kernel
        // path skips the F32 round-trip through HBM. The candle path is the
        // parity oracle exercised by `kiln-rmsnorm-kernel`'s
        // `parity_l2_qk_norm_*` tests.
        let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
        let defer_cuda_qk_norm_to_recurrent = {
            #[cfg(feature = "cuda")]
            {
                seq_len == 1
                    && gdn_forward_only_fastpaths
                    && !capture_b11_taps
                    && !capture_c41_taps
                    && fused_decode_unexpanded_qk
                    && input_dtype == DType::BF16
                    && backend.supports_gdn_decode_qk_norm_gates_recurrent()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        let defer_native_qk_norm_to_recurrent = seq_len == 1
            && gdn_forward_only_fastpaths
            && !capture_b11_taps
            && !capture_c41_taps
            && recurrent_unexpanded_qk
            && input_dtype == DType::BF16
            && backend.supports_gdn_recurrent_qk_norm_prefill_native_head_last();
        let (q, k, qk_expanded, qk_norm_deferred, qk_norm_deferred_to_native_recurrent) = {
            #[cfg(feature = "metal")]
            {
                if use_unexpanded_qk {
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_unexpanded");
                    let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
                    (q, k, false, false, false)
                } else if input_dtype == DType::BF16
                    && gdn_forward_only_fastpaths
                    && gqa_ratio > 1
                    && crate::backend::metal::metal_gdn_qk_norm_gqa_supports(&q, &k, nv)
                {
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_gqa");
                    crate::backend::metal::metal_gdn_qk_norm_gqa_f32_bf16(
                        &q,
                        &k,
                        nv,
                        scale as f32,
                        1e-6,
                    )
                    .context("metal gdn qk_norm gqa kernel failed")
                    .map(|(q, k)| (q, k, true, false, false))?
                } else {
                    let (q, k) = {
                        kiln_nvtx::range!(c"kiln/gdn/head_expand");
                        if gqa_ratio > 1 {
                            let q = q
                                .unsqueeze(3)?
                                .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
                                .contiguous()?
                                .reshape((batch, seq_len, nv, dk))?;
                            let k = k
                                .unsqueeze(3)?
                                .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
                                .contiguous()?
                                .reshape((batch, seq_len, nv, dk))?;
                            (q, k)
                        } else {
                            (q.contiguous()?, k.contiguous()?)
                        }
                    };
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm");
                    let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
                    (q, k, true, false, false)
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                let fused_gqa = {
                    #[cfg(feature = "cuda")]
                    {
                        let disabled = std::env::var("KILN_DISABLE_FUSED_L2_QK_NORM").is_ok();
                        if !fused_decode_unexpanded_qk
                            && gdn_forward_only_fastpaths
                            && !disabled
                            && input_dtype == DType::BF16
                            && gqa_ratio > 1
                            && kiln_rmsnorm_kernel::supports_l2_qk_norm_gqa(&q, &k, nv)
                        {
                            kiln_nvtx::range!(c"kiln/gdn/qk_norm_gqa");
                            Some(
                                kiln_rmsnorm_kernel::fused_l2_qk_norm_gqa(
                                    &q,
                                    &k,
                                    nv,
                                    scale as f32,
                                    1e-6,
                                )
                                .context("fused_l2_qk_norm_gqa kernel failed")?,
                            )
                        } else {
                            None
                        }
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        None
                    }
                };

                if defer_cuda_qk_norm_to_recurrent {
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_deferred");
                    (q, k, false, true, false)
                } else if defer_native_qk_norm_to_recurrent {
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_deferred_native");
                    (q, k, false, false, true)
                } else if fused_decode_unexpanded_qk {
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm_unexpanded");
                    let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
                    (q, k, false, false, false)
                } else if let Some((q, k)) = fused_gqa {
                    (q, k, true, false, false)
                } else {
                    let (q, k) = {
                        kiln_nvtx::range!(c"kiln/gdn/head_expand");
                        if gqa_ratio > 1 {
                            let q = q
                                .unsqueeze(3)?
                                .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
                                .contiguous()?
                                .reshape((batch, seq_len, nv, dk))?;
                            let k = k
                                .unsqueeze(3)?
                                .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
                                .contiguous()?
                                .reshape((batch, seq_len, nv, dk))?;
                            (q, k)
                        } else {
                            (q.contiguous()?, k.contiguous()?)
                        }
                    };
                    kiln_nvtx::range!(c"kiln/gdn/qk_norm");
                    let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
                    (q, k, true, false, false)
                }
            }
        };
        finish_gdn_stage_profile(
            profile_device,
            profile_context,
            "qk_norm",
            seq_len,
            stage_profile,
        )?;
        (
            q,
            k,
            v,
            z,
            qk_expanded,
            qk_norm_deferred,
            qk_norm_deferred_to_native_recurrent,
        )
    };

    // Phase B11b taps: `gdn_qk_norm_q` / `gdn_qk_norm_k`. Both are post-L2
    // normalization (+ Q scaled by 1/sqrt(dk)). Shapes [B, T, nv, dk] (the
    // GQA head-expand above brought nk→nv). HF mirror: `query` / `key` after
    // `query.normalize(dim=-1)` / `key.normalize(dim=-1)` and the Q-scale.
    if capture_b11_taps && qk_expanded {
        crate::mtp_debug::capture_b11_layer0_tap("gdn_qk_norm_q", &q)?;
        crate::mtp_debug::capture_b11_layer0_tap("gdn_qk_norm_k", &k)?;
    }
    if capture_c41_taps && qk_expanded {
        crate::mtp_debug::capture_c41_layer1_tap("gdn_qk_norm_q", &q)?;
        crate::mtp_debug::capture_c41_layer1_tap("gdn_qk_norm_k", &k)?;
    }

    // --- Step 7: Chunkwise analytical recurrence (Phase 6, approach (b)) ---
    // The recurrent state is stored in F32 externally (across layers/steps)
    // for accumulator stability, but we run the recurrence in bf16 to reclaim
    // the ~66% of prefill GPU time previously spent in bmul_f32 /
    // fast_sum_f32 / badd_f32 (see PROFILING.md recommendation #2). State is
    // cast to bf16 at entry and restored to F32 at exit so the external
    // invariant holds.
    //
    // PR #72 introduced the bf16 hot path. PR #74 replaced the read/write
    // broadcast_mul+sum pairs with batched matmuls but left the O(T)
    // sequential chain. This PR (Phase 6) unrolls the per-chunk recurrence
    // analytically: within each C = GDN_CHUNK_SIZE chunk we build a
    // triangular decay matrix and solve for the per-token updates in a small
    // number of heavy matmuls, cutting the number of GPU kernel launches
    // from O(T) to O(T / C) per layer.
    //
    // The within-chunk forward substitution still walks token-by-token, but
    // each step only does a [1, t] @ [t, dv] matmul over the already-built
    // prefix — orders of magnitude cheaper than the full [dk, dv] state
    // update that was previously done per token.
    let state_external_dtype = recurrent_state.dtype();
    if state_external_dtype != input_dtype {
        *recurrent_state = recurrent_state.to_dtype(input_dtype)?;
    }

    let fused_decode_gates_recurrent_rmsnorm = {
        let mut fused = {
            #[cfg(feature = "metal")]
            {
                if recurrent_unexpanded_qk
                    && seq_len == 1
                    && !capture_b11_taps
                    && !capture_c41_taps
                    && crate::backend::metal::metal_gdn_decode_gates_recurrent_rmsnorm_supports(
                        &q,
                        &k,
                        &v,
                        &a,
                        &b,
                        &weights.a_log,
                        &weights.dt_bias,
                        recurrent_state,
                        &z,
                        &weights.norm,
                    )
                {
                    kiln_nvtx::range!(c"kiln/gdn/gates_recur_gated_norm");
                    let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
                    let out = crate::backend::metal::metal_gdn_decode_gates_recurrent_rmsnorm_bf16(
                        &q,
                        &k,
                        &v,
                        &a,
                        &b,
                        &weights.a_log,
                        &weights.dt_bias,
                        recurrent_state,
                        &z,
                        &weights.norm,
                        config.rms_norm_eps as f32,
                    )
                    .context("metal gdn decode gates+recurrent+gated-rmsnorm kernel failed")?;
                    finish_gdn_stage_profile(
                        profile_device,
                        profile_context,
                        "gates_recur_gated_norm",
                        seq_len,
                        stage_profile,
                    )?;
                    Some(out)
                } else {
                    None
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                None
            }
        };
        if fused.is_none() && qk_norm_deferred_to_recurrent {
            kiln_nvtx::range!(c"kiln/gdn/qk_norm_gates_recur_gated_norm");
            let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
            fused = backend.gdn_decode_qk_norm_gates_recurrent_rmsnorm(
                &q,
                &k,
                &v,
                &a,
                &b,
                &weights.a_log_gates,
                &weights.dt_bias,
                recurrent_state,
                &z,
                &weights.norm,
                scale,
                1e-6,
                config.rms_norm_eps,
            )?;
            finish_gdn_stage_profile(
                profile_device,
                profile_context,
                "qk_norm_gates_recur_gated_norm",
                seq_len,
                stage_profile,
            )?;
        }
        fused
    };

    let fused_decode_gates_recurrent = {
        if fused_decode_gates_recurrent_rmsnorm.is_none()
            && gdn_forward_only_fastpaths
            && seq_len == 1
            && !capture_b11_taps
            && !capture_c41_taps
        {
            if qk_norm_deferred_to_recurrent {
                kiln_nvtx::range!(c"kiln/gdn/qk_norm_gates_recur");
                let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
                let out = if let Some(out) = backend.gdn_decode_qk_norm_gates_recurrent(
                    &q,
                    &k,
                    &v,
                    &a,
                    &b,
                    &weights.a_log_gates,
                    &weights.dt_bias,
                    recurrent_state,
                    scale,
                    1e-6,
                )? {
                    out
                } else {
                    let (q, k) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
                    backend
                        .gdn_decode_gates_recurrent(
                            &q,
                            &k,
                            &v,
                            &a,
                            &b,
                            &weights.a_log_gates,
                            &weights.dt_bias,
                            recurrent_state,
                            &z,
                            &weights.norm,
                            config.rms_norm_eps,
                        )?
                        .context("CUDA deferred qk_norm fallback recurrent path declined")?
                };
                finish_gdn_stage_profile(
                    profile_device,
                    profile_context,
                    "qk_norm_gates_recur",
                    seq_len,
                    stage_profile,
                )?;
                Some(out)
            } else if let Some(out) = backend.gdn_decode_gates_recurrent(
                &q,
                &k,
                &v,
                &a,
                &b,
                &weights.a_log_gates,
                &weights.dt_bias,
                recurrent_state,
                &z,
                &weights.norm,
                config.rms_norm_eps,
            )? {
                kiln_nvtx::range!(c"kiln/gdn/gates_recur");
                Some(out)
            } else {
                #[cfg(feature = "metal")]
                {
                    if recurrent_unexpanded_qk
                        && crate::backend::metal::metal_gdn_decode_gates_recurrent_supports(
                            &q,
                            &k,
                            &v,
                            &a,
                            &b,
                            &weights.a_log,
                            &weights.dt_bias,
                            recurrent_state,
                        )
                    {
                        kiln_nvtx::range!(c"kiln/gdn/gates_recur");
                        let stage_profile =
                            start_gdn_stage_profile(profile_device, profile_context)?;
                        let out = crate::backend::metal::metal_gdn_decode_gates_recurrent_bf16(
                            &q,
                            &k,
                            &v,
                            &a,
                            &b,
                            &weights.a_log,
                            &weights.dt_bias,
                            recurrent_state,
                        )
                        .context("metal gdn decode gates+recurrent kernel failed")?;
                        finish_gdn_stage_profile(
                            profile_device,
                            profile_context,
                            "gates_recur",
                            seq_len,
                            stage_profile,
                        )?;
                        Some(out)
                    } else {
                        None
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    None
                }
            }
        } else {
            None
        }
    };

    let fused_prefill_decay_recurrent = {
        #[cfg(feature = "metal")]
        {
            if fused_decode_gates_recurrent_rmsnorm.is_none()
                && fused_decode_gates_recurrent.is_none()
                && recurrent_unexpanded_qk
                && seq_len > 1
                && use_fused_gdn_gates
                && crate::backend::metal::metal_gdn_gates_decay_supports(
                    &a,
                    &b,
                    &weights.a_log,
                    &weights.dt_bias,
                )
            {
                let v_recur = v.to_dtype(input_dtype)?;
                if crate::backend::metal::metal_gdn_recurrent_prefill_native_head_last_decay_supports(
                    &q,
                    &k,
                    &v_recur,
                    &a,
                    &b,
                    recurrent_state,
                ) {
                    let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
                    let (beta, decay) = {
                        kiln_nvtx::range!(c"kiln/gdn/gates");
                        if let Some(ab) = prefill_ab_for_gates.as_ref() {
                            if crate::backend::metal::metal_gdn_gates_decay_ab_supports(
                                ab,
                                &weights.a_log,
                                &weights.dt_bias,
                                nv,
                            ) {
                                crate::backend::metal::metal_gdn_gates_decay_ab_bf16(
                                    ab,
                                    &weights.a_log,
                                    &weights.dt_bias,
                                    nv,
                                )
                                .context("metal gdn prefill A/B gates decay kernel failed")?
                            } else {
                                crate::backend::metal::metal_gdn_gates_decay_bf16(
                                    &a,
                                    &b,
                                    &weights.a_log,
                                    &weights.dt_bias,
                                )
                                .context("metal gdn prefill gates decay kernel failed")?
                            }
                        } else {
                            crate::backend::metal::metal_gdn_gates_decay_bf16(
                                &a,
                                &b,
                                &weights.a_log,
                                &weights.dt_bias,
                            )
                            .context("metal gdn prefill gates decay kernel failed")?
                        }
                    };
                    finish_gdn_stage_profile(
                        profile_device,
                        profile_context,
                        "gates",
                        seq_len,
                        stage_profile,
                    )?;

                    kiln_nvtx::range!(c"kiln/gdn/recurrent");
                    let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
                    let attn_out =
                        crate::backend::metal::metal_gdn_recurrent_prefill_native_head_last_decay_bf16(
                            &q,
                            &k,
                            &v_recur,
                            &beta,
                            &decay,
                            recurrent_state,
                        )
                        .context("metal gdn prefill recurrent decay kernel failed")?;
                    finish_gdn_stage_profile(
                        profile_device,
                        profile_context,
                        "recurrent",
                        seq_len,
                        stage_profile,
                    )?;
                    Some(attn_out)
                } else {
                    None
                }
            } else {
                None
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            None
        }
    };

    let expanded_qk_for_split = |q: Tensor, k: Tensor| -> Result<(Tensor, Tensor)> {
        if qk_expanded {
            Ok((q, k))
        } else {
            kiln_nvtx::range!(c"kiln/gdn/head_expand_recur_fallback");
            let q = q
                .unsqueeze(3)?
                .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
                .contiguous()?
                .reshape((batch, seq_len, nv, dk))?;
            let k = k
                .unsqueeze(3)?
                .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
                .contiguous()?
                .reshape((batch, seq_len, nv, dk))?;
            Ok((q, k))
        }
    };

    let (attn_out, attn_out_head_last, attn_out_already_gated_norm) = if let Some(attn_out) =
        fused_decode_gates_recurrent_rmsnorm
    {
        (attn_out, true, true) // [B, T, nv, dv], contiguous and gated-normalized
    } else if let Some(attn_out) = fused_decode_gates_recurrent {
        (attn_out, true, false) // [B, T, nv, dv], contiguous
    } else if let Some(attn_out) = fused_prefill_decay_recurrent {
        (attn_out, true, false) // [B, T, nv, dv], contiguous
    } else {
        // --- Step 6: Compute gates ---
        //
        // Two paths: a fused backend kernel (`backend.gdn_gates`) that collapses
        // the sigmoid + softplus + exp + mul chain into one launch, and the
        // candle-op reference path for everything outside the kernel's
        // envelope (unsupported backend, non-bf16, nv > 256, or kill switches
        // like `KILN_DISABLE_FUSED_GDN_GATES=1` /
        // `KILN_DISABLE_METAL_GDN_GATES=1`). The two are algorithmically
        // identical — the reference path is the original Phase-6 implementation
        // and remains the parity oracle.
        let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
        let (beta, g) = {
            kiln_nvtx::range!(c"kiln/gdn/gates");
            if gdn_forward_only_fastpaths && use_fused_gdn_gates && backend.supports_gdn_gates() {
                if let Some((beta, g)) = backend
                    .gdn_gates(&a, &b, &weights.a_log_gates, &weights.dt_bias)
                    .context("gdn decode gates fused backend")?
                {
                    (beta, g)
                } else {
                    gated_deltanet_gates_fallback(&a, &b, weights, input_dtype)
                        .context("gdn decode gates fallback after backend miss")?
                }
            } else {
                gated_deltanet_gates_fallback(&a, &b, weights, input_dtype)
                    .context("gdn decode gates fallback")?
            }
        };
        finish_gdn_stage_profile(
            profile_device,
            profile_context,
            "gates",
            seq_len,
            stage_profile,
        )?;

        // Phase B11b taps: `gdn_gate_beta` = sigmoid(b), `gdn_gate_g` =
        // -exp(A_log) * softplus(a + dt_bias) (the log-decay scalar fed into the
        // recurrence). Shapes [B, T, nv]. HF mirror: `beta = b.sigmoid()` and
        // `g = -A_log.exp() * F.softplus(a + dt_bias)`.
        if capture_b11_taps {
            crate::mtp_debug::capture_b11_layer0_tap("gdn_gate_beta", &beta)?;
            crate::mtp_debug::capture_b11_layer0_tap("gdn_gate_g", &g)?;
        }
        if capture_c41_taps {
            crate::mtp_debug::capture_c41_layer1_tap("gdn_gate_beta", &beta)?;
            crate::mtp_debug::capture_c41_layer1_tap("gdn_gate_g", &g)?;
        }

        let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
        let native_recurrent_result = if qk_norm_deferred_to_native_recurrent {
            let v_recur = v.to_dtype(input_dtype)?;
            match backend.gdn_recurrent_qk_norm_prefill_native_head_last(
                &q,
                &k,
                &v_recur,
                &beta,
                &g,
                recurrent_state,
                scale,
                1e-6,
            )? {
                Some(attn_out) => Some(attn_out),
                None => {
                    let (q_norm, k_norm) = gdn_qk_norm(&q, &k, input_dtype, scale)?;
                    let Some(attn_out) = gdn_recurrent_prefill_native_head_last(
                        backend,
                        &q_norm,
                        &k_norm,
                        &v_recur,
                        &beta,
                        &g,
                        recurrent_state,
                    )?
                    else {
                        anyhow::bail!(
                            "backend declined GDN qk-norm recurrent fallback after qk_norm deferral"
                        );
                    };
                    Some(attn_out)
                }
            }
        } else if recurrent_unexpanded_qk {
            let v_recur = v.to_dtype(input_dtype)?;
            gdn_recurrent_prefill_native_head_last(
                backend,
                &q,
                &k,
                &v_recur,
                &beta,
                &g,
                recurrent_state,
            )?
        } else {
            None
        };

        let recurrent_result = if let Some(attn_out) = native_recurrent_result {
            (attn_out, true, false) // [B, T, nv, dv], contiguous
        } else {
            let (q, k) = expanded_qk_for_split(q, k)?;

            // Cast v back to input_dtype so the recurrence stays in bf16. The
            // portable F32 causal-conv fallback can still produce F32 mixed_qkv;
            // without this cast the subtract `(v - exp(G) * (K @ S_entry))` below
            // hits a dtype mismatch on bf16 GPU runs, because the state-derived
            // tensor inherits the (now bf16) state dtype.
            let (q, k, v, beta, g) = {
                kiln_nvtx::range!(c"kiln/gdn/recur_prep");
                let v = v.to_dtype(input_dtype)?;

                // Transpose to [B, nv, T, dim] for per-head processing.
                let q = q.transpose(1, 2)?; // [B, nv, T, dk]
                let k = k.transpose(1, 2)?; // [B, nv, T, dk]
                let v = v.transpose(1, 2)?; // [B, nv, T, dv]
                let beta = beta.transpose(1, 2)?; // [B, nv, T]
                let g = g.transpose(1, 2)?; // [B, nv, T]
                (q, k, v, beta, g)
            };

            if allow_prefill_recurrent_kernel
                && let Some(attn_out) = gdn_recurrent_prefill_head_last(
                    backend,
                    &q,
                    &k,
                    &v,
                    &beta,
                    &g,
                    recurrent_state,
                )?
            {
                (attn_out, true, false) // [B, T, nv, dv], contiguous
            } else {
                match gdn_chunkwise_recurrence_head_last_full_chunks(
                    backend,
                    &q,
                    &k,
                    &v,
                    &beta,
                    &g,
                    recurrent_state,
                    GDN_CHUNK_SIZE,
                )? {
                    Some(attn_out) => (attn_out, true, false), // [B, T, nv, dv], contiguous
                    None => (
                        gdn_chunkwise_recurrence(
                            backend,
                            &q,
                            &k,
                            &v,
                            &beta,
                            &g,
                            recurrent_state,
                            GDN_CHUNK_SIZE,
                        )?,
                        false,
                        false,
                    ), // [B, nv, T, dv]
                }
            }
        };
        finish_gdn_stage_profile(
            profile_device,
            profile_context,
            "recurrent",
            seq_len,
            stage_profile,
        )?;
        recurrent_result
    };

    // Restore state to its original dtype so the caller's F32 invariant holds
    // across layer calls and across prefill/decode steps.
    if state_external_dtype != input_dtype {
        *recurrent_state = recurrent_state.to_dtype(state_external_dtype)?;
    }

    // Transpose to [B, T, nv, dv] unless the Metal full-chunk path already
    // wrote that contiguous layout directly.
    let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
    let attn_out = {
        kiln_nvtx::range!(c"kiln/gdn/post_transpose");
        if attn_out_head_last {
            attn_out
        } else {
            attn_out.transpose(1, 2)?
        }
    };
    finish_gdn_stage_profile(
        profile_device,
        profile_context,
        "post_transpose",
        seq_len,
        stage_profile,
    )?;

    // Phase B11b tap: `gdn_recur_out`. Captured post-transpose (shape
    // [B, T, nv, dv]) so the layout matches the input HF passes to its
    // GatedRMSNorm — i.e. the recurrence output transposed into the
    // "head-last" layout. Capturing here (rather than pre-transpose) lets
    // the HF reference mirror this tensor via a single
    // `norm.register_forward_pre_hook`, which sees exactly the same shape.
    if capture_b11_taps {
        crate::mtp_debug::capture_b11_layer0_tap("gdn_recur_out", &attn_out)?;
    }
    if capture_c41_taps {
        crate::mtp_debug::capture_c41_layer1_tap("gdn_recur_out", &attn_out)?;
    }

    // --- Step 8: Gated RMSNorm — norm(attn_out) * silu(z) ---
    let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
    let attn_out = {
        kiln_nvtx::range!(c"kiln/gdn/gated_norm");
        let attn_out = if attn_out_already_gated_norm {
            attn_out
        } else {
            gated_rms_norm(backend, &attn_out, &z, &weights.norm, config.rms_norm_eps)?
        };
        // Reshape to [B, T, v_dim] and cast back to input dtype
        attn_out
            .reshape((batch, seq_len, v_dim))?
            .to_dtype(input_dtype)?
    };
    finish_gdn_stage_profile(
        profile_device,
        profile_context,
        "gated_norm",
        seq_len,
        stage_profile,
    )?;

    // Phase B11b tap: `gdn_gated_norm`. Output of the GatedRMSNorm /
    // `norm(attn_out) * silu(z)` block, reshaped and cast back to input
    // dtype. Shape [B, T, v_dim]. HF mirror: `core_attn_out` after
    // `self.norm(core_attn_out, z)`.
    if capture_b11_taps {
        crate::mtp_debug::capture_b11_layer0_tap("gdn_gated_norm", &attn_out)?;
    }
    if capture_c41_taps {
        crate::mtp_debug::capture_c41_layer1_tap("gdn_gated_norm", &attn_out)?;
    }

    // --- Step 9: Output projection ---
    // NOTE: conv1d bias is not loaded by the weight loader. If the model has one,
    // it should be added to GpuLinearAttentionWeights and applied after conv1d.
    // Pre-transposed cache (see Step 1 note).
    let stage_profile = start_gdn_stage_profile(profile_device, profile_context)?;
    let out = {
        kiln_nvtx::range!(c"kiln/gdn/out_proj");
        mlp_proj_forward_decode_if(
            Some(backend),
            use_metal_decode_gemv,
            &attn_out,
            &weights.out_proj_t,
            weights.out_proj_marlin.as_ref(),
            gdn_out_lora,
            lora_scale,
        )?
    };
    finish_gdn_stage_profile(
        profile_device,
        profile_context,
        "out_proj",
        seq_len,
        stage_profile,
    )?;

    // Phase B11b tap: `gdn_out_proj`. Output of the final `out_proj` linear
    // (shape [B, T, hidden]) — this is what the caller adds to the residual
    // stream. HF mirror: `self.out_proj(core_attn_out)`.
    if capture_b11_taps {
        crate::mtp_debug::capture_b11_layer0_tap("gdn_out_proj", &out)?;
    }
    if capture_c41_taps {
        crate::mtp_debug::capture_c41_layer1_tap("gdn_out_proj", &out)?;
    }

    Ok(out)
}

/// Grouped-Query Attention (GQA).
///
/// Computes scaled dot-product attention with fewer KV heads than Q heads.
/// Each group of `num_heads / num_kv_heads` query heads shares one KV head.
///
/// `x`: [batch, seq_len, hidden_size]
/// `attn_weights`: Q/K/V/O projection weights plus per-head RMSNorm weights
/// `positions`: position indices for RoPE (length = seq_len, absolute positions)
/// `num_heads`: number of query attention heads
/// `num_kv_heads`: number of key/value attention heads
/// `head_dim`: dimension per head
/// `inv_freq`: cached RoPE frequency table (built once via [`compute_rotary_inv_freq`])
/// `rms_norm_eps`: epsilon for Q/K head norms
/// `kv_cache`: optional KV cache for incremental decoding
/// `full_attn_layer_idx`: index into the KV cache's layer array (only full-attn layers)
///
/// Dispatch `q_proj` through the Marlin W4A16 path when available, else the
/// existing BF16 `broadcast_matmul(q_proj_t)` path. LoRA deltas are always
/// added after the base matmul so behaviour matches `linear_with_lora_t` in
/// the absence of Marlin weights.
pub fn q_proj_forward(
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    q_proj_forward_decode_if(None, false, x, attn_weights, lora, lora_scale)
}

fn q_proj_forward_decode_if(
    backend: Option<&dyn BackendRuntime>,
    use_metal_decode_gemv: bool,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(ref packed) = attn_weights.q_proj_marlin {
        let base =
            crate::marlin_proj::matmul_bf16(x, packed).context("q_proj_forward: marlin matmul")?;
        if let Some(proj) = lora {
            let delta =
                compute_lora_delta(x, proj, lora_scale).context("q_proj_forward: lora delta")?;
            return Ok((base + delta).context("q_proj_forward: add lora delta")?);
        }
        return Ok(base);
    }
    linear_with_lora_t_backend_decode_if(
        backend,
        use_metal_decode_gemv,
        x,
        &attn_weights.q_proj_t,
        lora,
        lora_scale,
    )
}

#[cfg(feature = "cuda")]
fn cuda_split_q_gate_training_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| env_truthy("KILN_DISABLE_CUDA_SPLIT_Q_GATE_TRAINING"))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_split_q_gate_training_bf16(
    backend: &dyn BackendRuntime,
    use_metal_decode_gemv: bool,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    if cuda_split_q_gate_training_disabled()
        || !x.track_op()
        || x.dtype() != DType::BF16
        || !matches!(x.device(), Device::Cuda(_))
        || attn_weights.q_proj_marlin.is_some()
    {
        return Ok(None);
    }

    let q_dim = num_heads * head_dim;
    let Ok((_, q_out_dim)) = attn_weights.q_proj_t.dims2() else {
        return Ok(None);
    };
    if q_out_dim != q_dim * 2 {
        return Ok(None);
    }

    let q_weight_t = attn_weights.q_proj_t.narrow(1, 0, q_dim)?.contiguous()?;
    let gate_weight_t = attn_weights
        .q_proj_t
        .narrow(1, q_dim, q_dim)?
        .contiguous()?;

    let mut q_lora = None;
    let mut gate_lora = None;
    if let Some(proj) = lora {
        let Ok((b_out, _rank)) = proj.b.dims2() else {
            return Ok(None);
        };
        if b_out != q_dim * 2 {
            return Ok(None);
        }
        q_lora = Some(LoraProjectionWeights {
            a: proj.a.clone(),
            b: proj.b.narrow(0, 0, q_dim)?.contiguous()?,
        });
        gate_lora = Some(LoraProjectionWeights {
            a: proj.a.clone(),
            b: proj.b.narrow(0, q_dim, q_dim)?.contiguous()?,
        });
    }

    let q_flat = linear_with_lora_t_backend_decode_if(
        Some(backend),
        use_metal_decode_gemv,
        x,
        &q_weight_t,
        q_lora.as_ref(),
        lora_scale,
    )?;
    let gate = linear_with_lora_t_backend_decode_if(
        Some(backend),
        use_metal_decode_gemv,
        x,
        &gate_weight_t,
        gate_lora.as_ref(),
        lora_scale,
    )?;
    let q = q_flat.reshape(((), seq_len, num_heads, head_dim))?;
    let gate = gate.reshape(((), seq_len, q_dim))?;
    Ok(Some((q, gate)))
}

pub struct GqaAttentionPrepared {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub gate: Option<Tensor>,
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_q_gate_prefill(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &[u32],
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<(Tensor, Option<Tensor>)> {
    let (_batch, seq_len, _hidden) = x.dims3()?;
    let use_metal_decode_gemv = false;
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let split_q_gate = {
        #[cfg(feature = "cuda")]
        {
            if attn_output_gate {
                cuda_split_q_gate_training_bf16(
                    backend,
                    use_metal_decode_gemv,
                    x,
                    attn_weights,
                    lora_layer.and_then(|l| l.q_proj.as_ref()),
                    lora_scale,
                    seq_len,
                    num_heads,
                    head_dim,
                )?
            } else {
                None
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    };

    let (q, gate) = if let Some((q, gate)) = split_q_gate {
        (q, Some(gate))
    } else {
        let q_raw = q_proj_forward_decode_if(
            Some(backend),
            use_metal_decode_gemv,
            x,
            attn_weights,
            lora_layer.and_then(|l| l.q_proj.as_ref()),
            lora_scale,
        )?;
        if attn_output_gate {
            let q_raw = q_raw.reshape(((), seq_len, num_heads, head_dim * 2))?;
            let q = q_raw.narrow(3, 0, head_dim)?;
            let gate = q_raw.narrow(3, head_dim, head_dim)?;
            let gate = gate
                .contiguous()?
                .reshape(((), seq_len, num_heads * head_dim))?;
            (q.contiguous()?, Some(gate))
        } else {
            (q_raw.reshape(((), seq_len, num_heads, head_dim))?, None)
        }
    };

    let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
    let (q, _) = rotary_embedding(&q, &q, positions, head_dim, rotary_dim, inv_freq)?;
    Ok((q, gate))
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_kv_prefill(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &[u32],
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<(Tensor, Tensor)> {
    let (_batch, seq_len, _hidden) = x.dims3()?;
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let k = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        x,
        &attn_weights.k_proj_t,
        lora_layer.and_then(|l| l.k_proj.as_ref()),
        lora_scale,
    )?
    .reshape(((), seq_len, num_kv_heads, head_dim))?;
    let v = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        x,
        &attn_weights.v_proj_t,
        lora_layer.and_then(|l| l.v_proj.as_ref()),
        lora_scale,
    )?
    .reshape(((), seq_len, num_kv_heads, head_dim))?;
    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;
    let (k, _) = rotary_embedding(&k, &k, positions, head_dim, rotary_dim, inv_freq)?;
    Ok((k, v))
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_prepare_prefill(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &[u32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<GqaAttentionPrepared> {
    let (_batch, seq_len, _hidden) = x.dims3()?;
    let use_metal_decode_gemv = false;
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let split_q_gate = {
        #[cfg(feature = "cuda")]
        {
            if attn_output_gate {
                cuda_split_q_gate_training_bf16(
                    backend,
                    use_metal_decode_gemv,
                    x,
                    attn_weights,
                    lora_layer.and_then(|l| l.q_proj.as_ref()),
                    lora_scale,
                    seq_len,
                    num_heads,
                    head_dim,
                )?
            } else {
                None
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    };

    let (q_raw, k, v) = {
        kiln_nvtx::range!(c"kiln/proj/qkv");
        if split_q_gate.is_some() {
            let k = linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                x,
                &attn_weights.k_proj_t,
                lora_layer.and_then(|l| l.k_proj.as_ref()),
                lora_scale,
            )?;
            let v = linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                x,
                &attn_weights.v_proj_t,
                lora_layer.and_then(|l| l.v_proj.as_ref()),
                lora_scale,
            )?;
            (None, k, v)
        } else {
            let (q_raw, k, v) = full_attn_qkv_proj_decode_if(
                backend,
                use_metal_decode_gemv,
                x,
                attn_weights,
                lora_layer,
                lora_scale,
            )?;
            (Some(q_raw), k, v)
        }
    };

    let (q, gate) = if let Some((q, gate)) = split_q_gate {
        (q, Some(gate))
    } else if attn_output_gate {
        let q_raw = q_raw
            .as_ref()
            .expect("q_raw is present when split_q_gate is inactive");
        let q_raw = q_raw.reshape(((), seq_len, num_heads, head_dim * 2))?;
        let q = q_raw.narrow(3, 0, head_dim)?;
        let gate = q_raw.narrow(3, head_dim, head_dim)?;
        let gate = gate
            .contiguous()?
            .reshape(((), seq_len, num_heads * head_dim))?;
        (q.contiguous()?, Some(gate))
    } else {
        let q_raw = q_raw
            .as_ref()
            .expect("q_raw is present when attention output gate is disabled");
        let q = q_raw.reshape(((), seq_len, num_heads, head_dim))?;
        (q, None)
    };

    let k = k.reshape(((), seq_len, num_kv_heads, head_dim))?;
    let v = v.reshape(((), seq_len, num_kv_heads, head_dim))?;
    let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;
    let (q, k) = rotary_embedding(&q, &k, positions, head_dim, rotary_dim, inv_freq)?;

    Ok(GqaAttentionPrepared { q, k, v, gate })
}

pub fn gqa_attention_core_prefill(
    backend: &dyn BackendRuntime,
    prepared: &GqaAttentionPrepared,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let (_batch, seq_len, _heads, _hd) = prepared.q.dims4()?;
    if seq_len > 1 && backend.supports_flash_attn_prefill() {
        let q = prepared.q.contiguous()?;
        let k = prepared.k.contiguous()?;
        let v = prepared.v.contiguous()?;
        #[cfg(feature = "cuda")]
        if let Some(attn_output) =
            cuda_flash_attention_training_bf16(&q, &k, &v, num_heads, num_kv_heads, head_dim)?
        {
            return Ok(attn_output.reshape(((), seq_len, num_heads * head_dim))?);
        }
        if let Some(attn_output) =
            flash_attention_forward(backend, &q, &k, &v, num_heads, num_kv_heads, head_dim)?
        {
            return Ok(attn_output);
        }
    }

    let q = prepared.q.transpose(1, 2)?.contiguous()?;
    let k = prepared.k.transpose(1, 2)?.contiguous()?;
    let v = prepared.v.transpose(1, 2)?.contiguous()?;
    let gqa_ratio = num_heads / num_kv_heads;
    let batch = k.dim(0)?;
    let (k, v) = if gqa_ratio > 1 {
        let k = k
            .unsqueeze(2)?
            .expand(&[batch, num_kv_heads, gqa_ratio, seq_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, seq_len, head_dim))?;
        let v = v
            .unsqueeze(2)?
            .expand(&[batch, num_kv_heads, gqa_ratio, seq_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, seq_len, head_dim))?;
        (k, v)
    } else {
        (k.contiguous()?, v.contiguous()?)
    };

    let scale = (head_dim as f64).sqrt();
    let attn_scores = q.broadcast_matmul(&k.t()?)?;
    let attn_scores = (attn_scores / scale)?;
    let attn_scores = apply_causal_mask_with_offset(&attn_scores, seq_len, seq_len, 0)?;
    let attn_weights_softmax = cuda_softmax_last_dim(&attn_scores)?;
    let attn_output = attn_weights_softmax.broadcast_matmul(&v)?;
    Ok(attn_output
        .transpose(1, 2)?
        .contiguous()?
        .reshape(((), seq_len, num_heads * head_dim))?)
}

pub fn gqa_attention_apply_output_gate(
    attn_output: Tensor,
    gate: Option<&Tensor>,
) -> Result<Tensor> {
    attention_output_gate_decode_if(false, attn_output, gate)
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_pre_o_chunked_prefill(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &[u32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
    tile_size: usize,
) -> Result<Tensor> {
    let (_batch, seq_len, _hidden) = x.dims3()?;
    if tile_size == 0 || tile_size >= seq_len {
        return gqa_attention_pre_o(
            backend,
            x,
            attn_weights,
            positions,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            inv_freq,
            rms_norm_eps,
            None,
            0,
            attn_output_gate,
            lora,
        );
    }

    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };

    let k = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        x,
        &attn_weights.k_proj_t,
        lora_layer.and_then(|l| l.k_proj.as_ref()),
        lora_scale,
    )
    .context("chunked full-attention pre-o k projection")?
    .reshape(((), seq_len, num_kv_heads, head_dim))
    .context("chunked full-attention pre-o k reshape")?;
    let v = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        x,
        &attn_weights.v_proj_t,
        lora_layer.and_then(|l| l.v_proj.as_ref()),
        lora_scale,
    )
    .context("chunked full-attention pre-o v projection")?
    .reshape(((), seq_len, num_kv_heads, head_dim))
    .context("chunked full-attention pre-o v reshape")?;
    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)
        .context("chunked full-attention pre-o k norm")?;
    let (k, _) = rotary_embedding(&k, &k, positions, head_dim, rotary_dim, inv_freq)
        .context("chunked full-attention pre-o k rotary")?;

    let mut output_tiles = Vec::with_capacity(seq_len.div_ceil(tile_size));
    let mut tile_start = 0usize;
    while tile_start < seq_len {
        let tile_len = (seq_len - tile_start).min(tile_size);
        let tile_end = tile_start + tile_len;

        let x_tile = x.narrow(1, tile_start, tile_len).with_context(|| {
            format!("chunked full-attention pre-o input tile [{tile_start}, {tile_end})")
        })?;
        let q_raw = q_proj_forward_decode_if(
            Some(backend),
            false,
            &x_tile,
            attn_weights,
            lora_layer.and_then(|l| l.q_proj.as_ref()),
            lora_scale,
        )
        .with_context(|| {
            format!("chunked full-attention pre-o q projection [{tile_start}, {tile_end})")
        })?;
        let (q_tile, gate_tile) = if attn_output_gate {
            let q_raw = q_raw
                .reshape(((), tile_len, num_heads, head_dim * 2))
                .with_context(|| {
                    format!(
                        "chunked full-attention pre-o q/gate reshape [{tile_start}, {tile_end})"
                    )
                })?;
            let q = q_raw
                .narrow(3, 0, head_dim)
                .with_context(|| {
                    format!("chunked full-attention pre-o q split [{tile_start}, {tile_end})")
                })?
                .contiguous()
                .context("chunked full-attention pre-o q contiguous")?;
            let gate = q_raw
                .narrow(3, head_dim, head_dim)
                .with_context(|| {
                    format!("chunked full-attention pre-o gate split [{tile_start}, {tile_end})")
                })?
                .contiguous()
                .context("chunked full-attention pre-o gate contiguous")?
                .reshape(((), tile_len, num_heads * head_dim))
                .context("chunked full-attention pre-o gate reshape")?;
            (q, Some(gate))
        } else {
            (
                q_raw
                    .reshape(((), tile_len, num_heads, head_dim))
                    .with_context(|| {
                        format!("chunked full-attention pre-o q reshape [{tile_start}, {tile_end})")
                    })?,
                None,
            )
        };
        let q_tile = rms_norm(&q_tile, &attn_weights.q_norm, rms_norm_eps)
            .context("chunked full-attention pre-o q norm")?;
        let tile_positions = &positions[tile_start..tile_end];
        let (q_tile, _) = rotary_embedding(
            &q_tile,
            &q_tile,
            tile_positions,
            head_dim,
            rotary_dim,
            inv_freq,
        )
        .with_context(|| {
            format!("chunked full-attention pre-o q rotary [{tile_start}, {tile_end})")
        })?;
        let k_prefix = k.narrow(1, 0, tile_end).with_context(|| {
            format!("chunked full-attention pre-o k prefix [0, {tile_end}) for tile {tile_start}")
        })?;
        let v_prefix = v.narrow(1, 0, tile_end).with_context(|| {
            format!("chunked full-attention pre-o v prefix [0, {tile_end}) for tile {tile_start}")
        })?;
        let tile_prepared = GqaAttentionPrepared {
            q: q_tile,
            k: k_prefix,
            v: v_prefix,
            gate: None,
        };
        let attn_core =
            gqa_attention_core_prefill(backend, &tile_prepared, num_heads, num_kv_heads, head_dim)
                .with_context(|| {
                    format!("chunked full-attention pre-o core tile [{tile_start}, {tile_end})")
                })?;
        let attn_output = gqa_attention_apply_output_gate(attn_core, gate_tile.as_ref())
            .with_context(|| {
                format!("chunked full-attention pre-o gate tile [{tile_start}, {tile_end})")
            })?;
        output_tiles.push(attn_output);

        tile_start = tile_end;
    }

    let output_refs: Vec<&Tensor> = output_tiles.iter().collect();
    Tensor::cat(&output_refs, 1).context("chunked full-attention pre-o cat")
}

/// Returns the gated attention value before the final output projection:
/// [batch, seq_len, num_heads * head_dim].
pub fn gqa_attention_pre_o(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &[u32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    kv_cache: Option<&mut KvCache>,
    full_attn_layer_idx: usize,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (_batch, seq_len, _hidden) = x.dims3()?;
    let profile_device = x.device();
    let profile_context = profile_full_attn_stages_enabled().then_some((
        full_attn_layer_idx,
        positions.first().copied().unwrap_or(0) as usize,
    ));
    let use_metal_decode_gemv = seq_len == 1
        && kv_cache.is_some()
        && !crate::mtp_debug::is_mtp_single_token_self_attn_armed();

    // Project to Q, K, V (with optional LoRA delta)
    // When attn_output_gate is true, q_proj outputs [Q, gate] fused:
    //   q_proj: [num_heads * head_dim * 2, hidden_size]
    //   Split into Q [num_heads, head_dim] and gate [num_heads, head_dim]
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let split_q_gate = {
        #[cfg(feature = "cuda")]
        {
            if attn_output_gate {
                cuda_split_q_gate_training_bf16(
                    backend,
                    use_metal_decode_gemv,
                    x,
                    attn_weights,
                    lora_layer.and_then(|l| l.q_proj.as_ref()),
                    lora_scale,
                    seq_len,
                    num_heads,
                    head_dim,
                )?
            } else {
                None
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    };

    let (q_raw, k, v) = {
        let stage_profile = start_named_full_attn_stage_profile(
            profile_device,
            profile_context,
            "qkv_proj",
            seq_len,
        )?;
        kiln_nvtx::range!(c"kiln/proj/qkv");
        let out = if split_q_gate.is_some() {
            let k = linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                x,
                &attn_weights.k_proj_t,
                lora_layer.and_then(|l| l.k_proj.as_ref()),
                lora_scale,
            )?;
            let v = linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                x,
                &attn_weights.v_proj_t,
                lora_layer.and_then(|l| l.v_proj.as_ref()),
                lora_scale,
            )?;
            (None, k, v)
        } else {
            let (q_raw, k, v) = full_attn_qkv_proj_decode_if(
                backend,
                use_metal_decode_gemv,
                x,
                attn_weights,
                lora_layer,
                lora_scale,
            )?;
            (Some(q_raw), k, v)
        };
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "qkv_proj",
            seq_len,
            stage_profile,
        )?;
        out
    };

    // Split Q and gate if output gate is enabled
    let (q, gate) = if let Some((q, gate)) = split_q_gate {
        (q, Some(gate))
    } else if attn_output_gate {
        let q_raw = q_raw
            .as_ref()
            .expect("q_raw is present when split_q_gate is inactive");
        // q_raw: [batch, seq_len, num_heads * head_dim * 2]
        // Reshape to [batch, seq_len, num_heads, head_dim * 2] then split
        let q_raw = q_raw.reshape(((), seq_len, num_heads, head_dim * 2))?;
        let q = q_raw.narrow(3, 0, head_dim)?;
        let gate = q_raw.narrow(3, head_dim, head_dim)?;
        // gate needs to be [batch, seq_len, num_heads * head_dim] for later
        let gate = gate
            .contiguous()?
            .reshape(((), seq_len, num_heads * head_dim))?;
        (q.contiguous()?, Some(gate))
    } else {
        let q_raw = q_raw
            .as_ref()
            .expect("q_raw is present when attention output gate is disabled");
        let q = q_raw.reshape(((), seq_len, num_heads, head_dim))?;
        (q, None)
    };

    // Reshape K, V to [batch, seq_len, num_heads, head_dim]
    let (k, v) = {
        let stage_profile = start_named_full_attn_stage_profile(
            profile_device,
            profile_context,
            "kv_reshape",
            seq_len,
        )?;
        let out = (
            k.reshape(((), seq_len, num_kv_heads, head_dim))?,
            v.reshape(((), seq_len, num_kv_heads, head_dim))?,
        );
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "kv_reshape",
            seq_len,
            stage_profile,
        )?;
        out
    };

    // Apply per-head RMSNorm to Q and K (Qwen3.5 uses QK-norm)
    // q_norm/k_norm are [head_dim] — broadcast over [batch, seq_len, num_heads, head_dim]
    let (q, k) = {
        let stage_profile = start_named_full_attn_stage_profile(
            profile_device,
            profile_context,
            "qk_norm",
            seq_len,
        )?;
        let out = (
            rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?,
            rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?,
        );
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "qk_norm",
            seq_len,
            stage_profile,
        )?;
        out
    };

    // Apply RoPE (positions are absolute, so cached tokens get correct embeddings)
    // Only rotate first rotary_dim dimensions; the rest pass through unchanged.
    let (q, k) = {
        let stage_profile =
            start_named_full_attn_stage_profile(profile_device, profile_context, "rope", seq_len)?;
        let out = rotary_embedding(&q, &k, positions, head_dim, rotary_dim, inv_freq)?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "rope",
            seq_len,
            stage_profile,
        )?;
        out
    };

    // Fused-attention path for prefill (seq_len > 1, no KV cache).
    // Takes [batch, seq_len, num_heads, head_dim] — the layout we already
    // have. When a KV cache is present we fall through to the naive path,
    // which handles the cache update and Q_len != KV_len masking correctly.
    // Backend declines (returns None) on dtype mismatch so non-BF16 configs
    // (e.g. tests on F32) transparently fall back to naive softmax+matmul.
    if seq_len > 1 && kv_cache.is_none() && backend.supports_flash_attn_prefill() {
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        #[cfg(feature = "cuda")]
        if let Some(attn_output) =
            cuda_flash_attention_training_bf16(&q, &k, &v, num_heads, num_kv_heads, head_dim)?
        {
            let attn_output = attn_output.reshape(((), seq_len, num_heads * head_dim))?;
            let attn_output = attention_output_gate_decode_if(false, attn_output, gate.as_ref())?;
            return Ok(attn_output);
        }
        if let Some(attn_output) =
            flash_attention_forward(backend, &q, &k, &v, num_heads, num_kv_heads, head_dim)?
        {
            let attn_output = attention_output_gate_decode_if(false, attn_output, gate.as_ref())?;
            return Ok(attn_output);
        }
    }

    // Transpose to [batch, heads, seq_len, head_dim] for naive attention
    let (q, k, v) = {
        let stage_profile = start_named_full_attn_stage_profile(
            profile_device,
            profile_context,
            "qkv_transpose",
            seq_len,
        )?;
        let out = (
            q.transpose(1, 2)?.contiguous()?,
            k.transpose(1, 2)?.contiguous()?,
            v.transpose(1, 2)?.contiguous()?,
        );
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "qkv_transpose",
            seq_len,
            stage_profile,
        )?;
        out
    };

    // If KV cache is provided, update it and use full cached K/V
    let (k, v, kv_len) = if let Some(cache) = kv_cache {
        let stage_profile = start_named_full_attn_stage_profile(
            profile_device,
            profile_context,
            "kv_cache_update",
            seq_len,
        )?;
        let (full_k, full_v) = cache
            .update(full_attn_layer_idx, &k, &v)
            .context("KV cache update failed")?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "kv_cache_update",
            seq_len,
            stage_profile,
        )?;
        let kv_len = full_k.dim(2)?;
        (full_k, full_v, kv_len)
    } else {
        (k, v, seq_len)
    };

    // GQA head expansion: repeat K/V to match Q head count
    let gqa_ratio = num_heads / num_kv_heads;
    let batch = k.dim(0)?;
    let (k, v) = if gqa_ratio > 1 {
        let stage_profile = start_named_full_attn_stage_profile(
            profile_device,
            profile_context,
            "gqa_expand",
            seq_len,
        )?;
        // Expand [batch, num_kv_heads, kv_len, head_dim] -> [batch, num_heads, kv_len, head_dim]
        let out = (
            k.unsqueeze(2)?
                .expand(&[batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
                .contiguous()?
                .reshape((batch, num_heads, kv_len, head_dim))?,
            v.unsqueeze(2)?
                .expand(&[batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
                .contiguous()?
                .reshape((batch, num_heads, kv_len, head_dim))?,
        );
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "gqa_expand",
            seq_len,
            stage_profile,
        )?;
        out
    } else {
        (k.contiguous()?, v.contiguous()?)
    };

    // Scaled dot-product attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
    // Q: [batch, num_heads, seq_len, head_dim]
    // K: [batch, num_heads, kv_len, head_dim]
    // scores: [batch, num_heads, seq_len, kv_len]
    let scale = (head_dim as f64).sqrt();
    let attn_scores = {
        let stage_profile = start_named_full_attn_stage_profile(
            profile_device,
            profile_context,
            "score_matmul",
            seq_len,
        )?;
        let out = q.broadcast_matmul(&k.t()?)?;
        let out = (out / scale)?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "score_matmul",
            seq_len,
            stage_profile,
        )?;
        out
    };

    // Apply causal mask (handles Q_len != KV_len for cached decoding)
    let past_len = kv_len - seq_len;
    let attn_scores = {
        let stage_profile = start_named_full_attn_stage_profile(
            profile_device,
            profile_context,
            "causal_mask",
            seq_len,
        )?;
        let out = apply_causal_mask_with_offset(&attn_scores, seq_len, kv_len, past_len)?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "causal_mask",
            seq_len,
            stage_profile,
        )?;
        out
    };

    let attn_weights_softmax = {
        let stage_profile = start_named_full_attn_stage_profile(
            profile_device,
            profile_context,
            "softmax",
            seq_len,
        )?;
        let out = cuda_softmax_last_dim(&attn_scores)?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "softmax",
            seq_len,
            stage_profile,
        )?;
        out
    };
    let attn_output = {
        let stage_profile = start_named_full_attn_stage_profile(
            profile_device,
            profile_context,
            "value_matmul",
            seq_len,
        )?;
        let out = attn_weights_softmax.broadcast_matmul(&v)?; // [batch, num_heads, seq_len, head_dim]
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "value_matmul",
            seq_len,
            stage_profile,
        )?;
        out
    };

    // Transpose back: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden]
    let attn_output = {
        let stage_profile = start_named_full_attn_stage_profile(
            profile_device,
            profile_context,
            "attn_output_layout",
            seq_len,
        )?;
        let out = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            (),
            seq_len,
            num_heads * head_dim,
        ))?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "attn_output_layout",
            seq_len,
            stage_profile,
        )?;
        out
    };

    let attn_output =
        attention_output_gate_decode_if(use_metal_decode_gemv, attn_output, gate.as_ref())?;
    Ok(attn_output)
}

pub fn gqa_attention_output_projection(
    backend: &dyn BackendRuntime,
    attn_output: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    use_metal_decode_gemv: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    kiln_nvtx::range!(c"kiln/proj/o");
    linear_with_lora_t_backend_decode_if(
        Some(backend),
        use_metal_decode_gemv,
        attn_output,
        &attn_weights.o_proj_t,
        lora_layer.and_then(|l| l.o_proj.as_ref()),
        lora_scale,
    )
}

/// Returns: [batch, seq_len, hidden_size]
pub fn gqa_attention(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &[u32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    kv_cache: Option<&mut KvCache>,
    full_attn_layer_idx: usize,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (_batch, seq_len, _hidden) = x.dims3()?;
    let use_metal_decode_gemv = seq_len == 1
        && kv_cache.is_some()
        && !crate::mtp_debug::is_mtp_single_token_self_attn_armed();
    let attn_output = gqa_attention_pre_o(
        backend,
        x,
        attn_weights,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        inv_freq,
        rms_norm_eps,
        kv_cache,
        full_attn_layer_idx,
        attn_output_gate,
        lora,
    )?;
    gqa_attention_output_projection(
        backend,
        &attn_output,
        attn_weights,
        use_metal_decode_gemv,
        lora,
    )
}

#[cfg(feature = "cuda")]
pub(crate) struct PagedDecodeGraphInputs<'a> {
    pub block_table: &'a Tensor,
    pub seqused_k: &'a Tensor,
    pub kv_slot: &'a Tensor,
    pub max_seqlen_k: usize,
    pub rotary_cos: &'a Tensor,
    pub rotary_sin: &'a Tensor,
    pub attn_out: &'a [Tensor],
    pub softmax_lse: &'a [Tensor],
}

/// Stable per-step inputs threaded through the batched CUDA graph
/// capture/replay path. Mirrors [`PagedDecodeGraphInputs`] but every
/// tensor is shaped for `[batch, …]`. The CUDA graph runner pre-allocates
/// these on the device once per `batch_size` bucket; per-step updates
/// rewrite their contents in place via `cudaMemcpyHtoDAsync` so the
/// captured kernels read from the same device pointers on every replay.
/// Not consumed yet — kept for the upcoming batched forward wrapper.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub(crate) struct BatchedPagedDecodeGraphInputs<'a> {
    /// `[batch]` u32 token-id buffer.
    pub token_ids: &'a Tensor,
    /// `[batch]` f32 per-row decode position.
    pub positions: &'a Tensor,
    /// `[batch, max_blocks_per_seq]` u32 padded block table.
    pub block_table: &'a Tensor,
    /// `[batch]` i32 per-row K/V length.
    pub seqused_k: &'a Tensor,
    /// `[batch]` u32 per-row current KV-write slot.
    pub kv_slot: &'a Tensor,
    /// Max K/V length baked into the captured kernel launch shape.
    pub max_seqlen_k: usize,
    /// `[batch, rotary_dim/2]` rotary cosine table.
    pub rotary_cos: &'a Tensor,
    /// `[batch, rotary_dim/2]` rotary sine table.
    pub rotary_sin: &'a Tensor,
    /// Per-full-attention-layer paged decode output buffers, shape
    /// `[batch, 1, n_heads, head_dim]`.
    pub attn_out: &'a [Tensor],
    /// Per-full-attention-layer paged decode LSE scratch, shape
    /// `[batch, n_heads, 1]`.
    pub softmax_lse: &'a [Tensor],
    /// `[batch, 1, vocab]` stable output-logits buffer. The captured
    /// forward writes the final logits into this storage via
    /// `slice_set` so replay always reads from the same device
    /// pointer; the runner argmax-reduces and DtoH-transfers tokens
    /// outside the captured region.
    pub output_logits: &'a Tensor,
    /// Persistent batched [`LinearAttentionState`] slot used by the
    /// captured forward. Lifetime is the graph runner's; the captured
    /// graph reads recurrent/conv state from these device pointers.
    pub linear_state: &'a mut LinearAttentionState,
}

/// Try the fused paged-decode flash-attention kernel.
///
/// Returns `Ok(Some(output))` on success and `Ok(None)` when the kernel
/// preconditions cannot be satisfied (forcing the caller to fall back to the
/// materializing slow path).
///
/// ### Preconditions checked here
///   * `block_size` divides `kBlockN = 128`
///   * Within each `kBlockN`-wide chunk of the block table, the underlying
///     physical pages are contiguous in the pool. The FA2 splitkv paged kernel
///     reads only one block-table entry per kBlockN chunk and assumes the next
///     `kBlockN / block_size` pages are physically contiguous (see
///     `flash_fwd_kernel.h` lines 587-596 and 770-779).
///
/// ### Output
/// `[batch, 1, num_heads * head_dim]` after o_proj (matches the slow path).
#[allow(clippy::too_many_arguments)]
fn try_flash_attn_paged_decode(
    backend: &dyn BackendRuntime,
    q: &Tensor,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    total_seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gate: Option<&Tensor>,
    use_metal_decode_gemv: bool,
    attn_weights: &GpuFullAttentionWeights,
    lora_layer: Option<&LoraLayerWeights>,
    lora_scale: f32,
    #[cfg(feature = "cuda")] graph_inputs: Option<&PagedDecodeGraphInputs<'_>>,
    profile_context: Option<(usize, usize)>,
) -> Result<Option<Tensor>> {
    const K_BLOCK_N: usize = 128;

    let block_size = paged_cache.block_size();
    if block_size == 0 || K_BLOCK_N % block_size != 0 {
        return Ok(None);
    }
    let pages_per_chunk = K_BLOCK_N / block_size;

    // q here is [batch, num_heads, 1, head_dim] after the transpose at the
    // call site. Flash-attn wants [batch, 1, num_heads, head_dim].
    let (batch, q_heads, q_len, q_hd) = q.dims4()?;
    if q_len != 1 || q_heads != num_heads || q_hd != head_dim {
        return Ok(None);
    }
    if batch != 1 {
        // Multi-sequence dispatch needs a per-sequence block_table tensor.
        // Defer to the slow path until the scheduler exercises it.
        return Ok(None);
    }

    let (k_pool, v_pool) = match paged_cache.pool_tensors(full_attn_layer_idx) {
        Some(p) => p,
        None => return Ok(None),
    };

    // Common macOS/desktop case: a single sequence receives freshly-allocated
    // blocks, so its whole live KV window is already one contiguous run in the
    // pool. In that case we can bypass the paged gather path entirely and feed
    // the fused prefill kernel a direct `[1, total_seq_len, kv_heads, head_dim]`
    // narrow of the live K/V window.
    // CUDA has a native GQA paged-decode kernel. The contiguous branch below
    // falls back to prefill attention on CUDA, expanding K/V heads and losing
    // time on the single-request decode path.
    let use_cuda_direct_paged_decode =
        backend.name() == "cuda" && !cuda_direct_paged_decode_disabled();
    if !paged_cache.is_fp8() && !use_cuda_direct_paged_decode {
        if let Some(start_slot) =
            contiguous_slot_run_start(block_table, block_size, 0, total_seq_len)
        {
            let softmax_scale = 1.0 / (head_dim as f32).sqrt();
            let stage_profile = start_full_attn_stage_profile(q.device(), profile_context)?;
            let attn_output = {
                kiln_nvtx::range!(c"kiln/attn/paged_decode_contiguous");
                backend.flash_attn_paged_decode_contiguous(
                    q,
                    k_pool,
                    v_pool,
                    start_slot,
                    total_seq_len,
                    softmax_scale,
                )?
            };
            finish_full_attn_stage_profile(
                q.device(),
                profile_context,
                "decode_attn_contiguous",
                q_len,
                stage_profile,
            )?;
            let attn_output = if attn_output.is_some() {
                attn_output
            } else {
                let fast_head_major = if backend.supports_flash_attn_prefill_head_major()
                    && backend.supports_paged_kv_head_major_read()
                {
                    kiln_nvtx::range!(c"kiln/kv/head_major_read_decode");
                    let stage_profile = start_full_attn_stage_profile(q.device(), profile_context)?;
                    let out = backend.paged_kv_head_major_read(
                        k_pool,
                        v_pool,
                        start_slot,
                        total_seq_len,
                    )?;
                    finish_full_attn_stage_profile(
                        q.device(),
                        profile_context,
                        "kv_head_read",
                        q_len,
                        stage_profile,
                    )?;
                    out
                } else {
                    None
                };
                if backend.supports_flash_attn_prefill_head_major() {
                    // Q is already head-major at the call site. Keep K/V grouped
                    // instead of routing through `flash_attention_forward`, which
                    // expands GQA K/V before Metal SDPA and defeats Candle's
                    // native vector-attention GQA path.
                    let (k_head, v_head) = match fast_head_major {
                        Some(kv) => kv,
                        None => {
                            let k_live =
                                k_pool.narrow(0, start_slot, total_seq_len)?.unsqueeze(0)?;
                            let v_live =
                                v_pool.narrow(0, start_slot, total_seq_len)?.unsqueeze(0)?;
                            (
                                k_live.transpose(1, 2)?.contiguous()?,
                                v_live.transpose(1, 2)?.contiguous()?,
                            )
                        }
                    };
                    let stage_profile = start_full_attn_stage_profile(q.device(), profile_context)?;
                    let out = flash_attention_forward_head_major(
                        backend, q, &k_head, &v_head, num_heads, head_dim,
                    )?;
                    finish_full_attn_stage_profile(
                        q.device(),
                        profile_context,
                        "decode_attn_head_major",
                        q_len,
                        stage_profile,
                    )?;
                    out
                } else {
                    None
                }
            };
            let attn_output = if attn_output.is_some() {
                attn_output
            } else {
                // Reshape Q for the fused-attention APIs only when the
                // head-major path declined. The common Metal desktop path
                // returns above and should not pay this transpose/copy.
                let k_live = k_pool.narrow(0, start_slot, total_seq_len)?.unsqueeze(0)?;
                let v_live = v_pool.narrow(0, start_slot, total_seq_len)?.unsqueeze(0)?;
                let stage_profile = start_full_attn_stage_profile(q.device(), profile_context)?;
                let q_fa = {
                    kiln_nvtx::range!(c"kiln/attn/q_fa_transpose");
                    q.transpose(1, 2)?.contiguous()?
                };
                finish_full_attn_stage_profile(
                    q.device(),
                    profile_context,
                    "q_fa_transpose",
                    q_len,
                    stage_profile,
                )?;
                let stage_profile = start_full_attn_stage_profile(q.device(), profile_context)?;
                let out = flash_attention_forward(
                    backend,
                    &q_fa,
                    &k_live,
                    &v_live,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )?;
                finish_full_attn_stage_profile(
                    q.device(),
                    profile_context,
                    "decode_attn_fallback",
                    q_len,
                    stage_profile,
                )?;
                out
            };
            if let Some(attn_output) = attn_output {
                // The flash-attention helpers already reshape to
                // [batch, seq_len, num_heads * head_dim].
                let _ = crate::mtp_debug::capture_subop("post_attn_raw", &attn_output);

                let stage_profile = start_full_attn_stage_profile(q.device(), profile_context)?;
                let attn_output =
                    attention_output_gate_decode_if(use_metal_decode_gemv, attn_output, gate)?;
                finish_full_attn_stage_profile(
                    q.device(),
                    profile_context,
                    "attn_gate",
                    q_len,
                    stage_profile,
                )?;
                let _ = crate::mtp_debug::capture_subop("post_attn_gated", &attn_output);

                let stage_profile = start_full_attn_stage_profile(q.device(), profile_context)?;
                let out = {
                    kiln_nvtx::range!(c"kiln/proj/o");
                    linear_with_lora_t_backend_decode_if(
                        Some(backend),
                        use_metal_decode_gemv,
                        &attn_output,
                        &attn_weights.o_proj_t,
                        lora_layer.and_then(|l| l.o_proj.as_ref()),
                        lora_scale,
                    )?
                };
                finish_full_attn_stage_profile(
                    q.device(),
                    profile_context,
                    "o_proj",
                    q_len,
                    stage_profile,
                )?;
                let _ = crate::mtp_debug::capture_subop("post_o_proj", &out);
                return Ok(Some(out));
            }
        }
    }

    // Verify intra-chunk contiguity. The kernel reads block_table[c * 8] only
    // (for block_size=16) and assumes pages [c*8 .. c*8+7] are physically
    // contiguous in the pool. kiln's `BlockManager` allocates blocks
    // sequentially from a free list, so a single freshly-allocated sequence
    // satisfies this trivially. After eviction or interleaved allocation the
    // condition may not hold, in which case we fall back.
    let n_chunks = total_seq_len.div_ceil(K_BLOCK_N);
    let blocks = &block_table.blocks;
    let allocated = blocks.len();
    if allocated < n_chunks * pages_per_chunk && allocated < total_seq_len.div_ceil(block_size) {
        // Block table too short for the requested seqlen.
        return Ok(None);
    }
    for c in 0..n_chunks {
        let base_idx = c * pages_per_chunk;
        if base_idx >= allocated {
            break;
        }
        let base_phys = blocks[base_idx];
        for i in 1..pages_per_chunk {
            let idx = base_idx + i;
            if idx >= allocated {
                break;
            }
            if blocks[idx] != base_phys + i as u32 {
                return Ok(None);
            }
        }
    }

    // Build a padded block_table tensor sized [1, n_chunks * pages_per_chunk].
    // Only the entries at indices c * pages_per_chunk are read by the kernel,
    // but we copy the active prefix of the kiln block table and pad the tail
    // by continuing the contiguous run from the last valid block (so any
    // stray reads stay within the cache pool).
    //
    // The scheduler may over-allocate blocks (blocks.len() > max_blocks_per_seq)
    // when it reserves capacity ahead of the current decode position. Those
    // extra blocks are not part of this iteration's active attention window,
    // so we truncate to max_blocks_per_seq before copying. Without this,
    // `reshape((1, max_blocks_per_seq))` crashes when allocated > max
    // (observed: 40 blocks vs max 32 at block 3 of full-attention layers).
    let max_blocks_per_seq = n_chunks * pages_per_chunk;
    let take = max_blocks_per_seq.min(blocks.len());
    let mut padded: Vec<u32> = Vec::with_capacity(max_blocks_per_seq);
    padded.extend_from_slice(&blocks[..take]);
    if padded.is_empty() {
        return Ok(None);
    }
    while padded.len() < max_blocks_per_seq {
        let next = padded.last().copied().unwrap_or(0).wrapping_add(1);
        padded.push(next);
    }

    let device = q.device();
    let bt_tensor_owned;
    let bt_tensor = {
        #[cfg(feature = "cuda")]
        {
            if let Some(inputs) = graph_inputs {
                inputs.block_table
            } else {
                bt_tensor_owned = Tensor::new(padded.as_slice(), device)?
                    .reshape((1usize, max_blocks_per_seq))?;
                &bt_tensor_owned
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            bt_tensor_owned =
                Tensor::new(padded.as_slice(), device)?.reshape((1usize, max_blocks_per_seq))?;
            &bt_tensor_owned
        }
    };

    let softmax_scale = 1.0f32 / (head_dim as f32).sqrt();

    // Reshape Q for the fused paged-decode APIs: [batch, num_heads, 1, head_dim]
    // -> [batch, 1, num_heads, head_dim]. Build it lazily so the contiguous-KV
    // Metal path above can avoid a dead transpose/copy per full-attention layer.
    let q_fa = {
        let stage_profile = start_full_attn_stage_profile(q.device(), profile_context)?;
        kiln_nvtx::range!(c"kiln/attn/q_fa_transpose");
        let q_fa = q.transpose(1, 2)?.contiguous()?;
        finish_full_attn_stage_profile(
            q.device(),
            profile_context,
            "q_fa_transpose",
            q_len,
            stage_profile,
        )?;
        q_fa
    };

    let stage_profile = start_full_attn_stage_profile(q.device(), profile_context)?;
    let attn_out = {
        #[cfg(feature = "cuda")]
        {
            if let Some(inputs) = graph_inputs {
                let attn_out = inputs.attn_out.get(full_attn_layer_idx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "missing CUDA graph paged decode output buffer for full-attention layer {full_attn_layer_idx}"
                    )
                })?;
                let softmax_lse = inputs.softmax_lse.get(full_attn_layer_idx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "missing CUDA graph paged decode LSE buffer for full-attention layer {full_attn_layer_idx}"
                    )
                })?;
                kiln_flash_attn::flash_attn_paged_decode_dyn_seqlen(
                    &q_fa,
                    k_pool,
                    v_pool,
                    bt_tensor,
                    inputs.seqused_k,
                    Some((attn_out, softmax_lse)),
                    inputs.max_seqlen_k,
                    block_size,
                    softmax_scale,
                    true,
                )?
            } else {
                match backend.flash_attn_paged_decode(
                    &q_fa,
                    k_pool,
                    v_pool,
                    bt_tensor,
                    total_seq_len,
                    block_size,
                    softmax_scale,
                    true,
                )? {
                    Some(t) => t,
                    None => return Ok(None),
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            match backend.flash_attn_paged_decode(
                &q_fa,
                k_pool,
                v_pool,
                bt_tensor,
                total_seq_len,
                block_size,
                softmax_scale,
                true,
            )? {
                Some(t) => t,
                None => return Ok(None),
            }
        }
    };
    finish_full_attn_stage_profile(
        q.device(),
        profile_context,
        "decode_attn_paged",
        q_len,
        stage_profile,
    )?;

    // attn_out is [batch, 1, num_heads, head_dim] bf16. Reshape to
    // [batch, 1, num_heads * head_dim] for the gate / o_proj path.
    let _ = num_kv_heads; // unused — kept in signature for symmetry / future use
    let attn_output = attn_out.reshape((batch, 1usize, num_heads * head_dim))?;
    let _ = crate::mtp_debug::capture_subop("post_attn_raw", &attn_output);

    let stage_profile = start_full_attn_stage_profile(q.device(), profile_context)?;
    let attn_output = attention_output_gate_decode_if(use_metal_decode_gemv, attn_output, gate)?;
    finish_full_attn_stage_profile(
        q.device(),
        profile_context,
        "attn_gate",
        q_len,
        stage_profile,
    )?;
    let _ = crate::mtp_debug::capture_subop("post_attn_gated", &attn_output);

    let stage_profile = start_full_attn_stage_profile(q.device(), profile_context)?;
    let out = {
        kiln_nvtx::range!(c"kiln/proj/o");
        linear_with_lora_t_backend_decode_if(
            Some(backend),
            use_metal_decode_gemv,
            &attn_output,
            &attn_weights.o_proj_t,
            lora_layer.and_then(|l| l.o_proj.as_ref()),
            lora_scale,
        )?
    };
    finish_full_attn_stage_profile(q.device(), profile_context, "o_proj", q_len, stage_profile)?;
    let _ = crate::mtp_debug::capture_subop("post_o_proj", &out);
    Ok(Some(out))
}

/// Per-decode-step metadata that is identical across every full-attention
/// layer (8× on Qwen3.5-4B). Building the `seqused_k` and padded
/// `block_table` tensors costs one `cudaMemcpyHtoD` each per build, and was
/// being repeated per layer — nsys at bs=16 attributed ~11% of GPU time to
/// these `copy2d_bf16` launches. Hoisted to once-per-step via this struct.
pub struct CachedPagedDecodeMeta {
    /// Padded `[batch, max_blocks_per_seq]` u32 tensor indexing the paged
    /// KV pool. Same for every full-attn layer within a step.
    pub block_table_tensor: Tensor,
    /// Per-row K/V length `[batch]` i32 tensor.
    pub seqused_k_tensor: Tensor,
    /// Max K/V length across rows in the batch (`max(start_pos) + 1`).
    pub max_seqlen_k: usize,
    /// Padded block-table width (in pages).
    pub max_blocks_per_seq: usize,
    /// Whether every row's `start_pos` is identical — when true, the strict
    /// uniform-length path is preferred over `dyn_seqlen`.
    pub uniform_start_pos: bool,
    /// When the uniform-length path is reachable, the per-row contiguous
    /// slot start positions (built via `paged_cache.contiguous_slot_run_starts`).
    /// Cached here so the strict fallback skips its own build too.
    pub strict_start_slots: Option<Vec<u32>>,
}

impl CachedPagedDecodeMeta {
    /// Build the shared metadata once for the current decode step. Mirrors
    /// the inline build inside `gqa_attention_paged_decode_contiguous_batch`,
    /// but yields tensors the caller can pass into every full-attn layer.
    pub fn build(
        device: &Device,
        paged_cache: &PagedKvCache,
        block_tables: &[&BlockTable],
        start_positions: &[usize],
    ) -> Result<Self> {
        let batch = start_positions.len();
        anyhow::ensure!(
            batch > 0,
            "CachedPagedDecodeMeta requires a non-empty batch"
        );
        anyhow::ensure!(
            block_tables.len() == batch,
            "CachedPagedDecodeMeta metadata length mismatch ({} vs {batch})",
            block_tables.len()
        );

        let max_start_pos = *start_positions
            .iter()
            .max()
            .context("CachedPagedDecodeMeta requires non-empty start_positions")?;
        let min_start_pos = *start_positions
            .iter()
            .min()
            .context("CachedPagedDecodeMeta requires non-empty start_positions")?;
        let uniform_start_pos = max_start_pos == min_start_pos;
        let max_seqlen_k = max_start_pos + 1;

        let page_block_size = paged_cache.block_size();
        let max_blocks_per_seq =
            ((max_seqlen_k + page_block_size - 1) / page_block_size).max(1);
        let mut block_table_vec = Vec::<u32>::with_capacity(batch * max_blocks_per_seq);
        let mut seqused_k_vec = Vec::<i32>::with_capacity(batch);
        for (row_idx, bt) in block_tables.iter().enumerate() {
            let row_seqlen = start_positions[row_idx] + 1;
            seqused_k_vec.push(
                i32::try_from(row_seqlen)
                    .context("CachedPagedDecodeMeta: seqused_k exceeds i32 range")?,
            );
            let row_blocks = bt.blocks.as_slice();
            anyhow::ensure!(
                row_blocks.len() * page_block_size >= row_seqlen,
                "CachedPagedDecodeMeta row {row_idx}: block_table covers {} tokens but row needs {}",
                row_blocks.len() * page_block_size,
                row_seqlen,
            );
            let pad_block = *row_blocks.last().unwrap_or(&0);
            for slot in 0..max_blocks_per_seq {
                let phys = if slot < row_blocks.len() {
                    row_blocks[slot]
                } else {
                    pad_block
                };
                block_table_vec.push(phys);
            }
        }

        let strict_start_slots: Option<Vec<u32>> = if uniform_start_pos {
            let live_window_starts = vec![0usize; batch];
            match paged_cache.contiguous_slot_run_starts(
                block_tables,
                &live_window_starts,
                max_seqlen_k,
            ) {
                Some(slots) => {
                    let v: Result<Vec<u32>> = slots
                        .iter()
                        .map(|&slot| {
                            u32::try_from(slot).context(
                                "CachedPagedDecodeMeta: start slot exceeds u32 range",
                            )
                        })
                        .collect();
                    Some(v?)
                }
                None => None,
            }
        } else {
            None
        };

        let block_table_tensor = Tensor::from_slice(
            block_table_vec.as_slice(),
            (batch, max_blocks_per_seq),
            device,
        )?
        .contiguous()?;
        let seqused_k_tensor =
            Tensor::from_slice(seqused_k_vec.as_slice(), batch, device)?.contiguous()?;

        Ok(Self {
            block_table_tensor,
            seqused_k_tensor,
            max_seqlen_k,
            max_blocks_per_seq,
            uniform_start_pos,
            strict_start_slots,
        })
    }
}

/// Batched full-attention decode for rows whose live paged-KV windows can be
/// addressed through a block table. Uniform contiguous rows use the strict
/// faster path when available; divergent row lengths use the backend's
/// dyn-seqlen path.
///
/// This is the scheduler-facing low-level primitive for true decode batching:
/// it projects Q/K/V for `[batch, 1, hidden]`, writes one K/V row per request
/// into the shared paged cache, runs the batched contiguous paged-attention
/// backend kernel, then applies the attention output gate and `o_proj`.
///
/// Current backend constraints are intentionally narrow:
/// - one decode token per row,
/// - non-FP8 paged cache,
/// - either each row's live `0..start_pos+1` KV window is one contiguous pool
///   run with a uniform length, or the backend accepts
///   `flash_attn_paged_decode_contiguous_batch_dyn_seqlen`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_paged_decode_contiguous_batch(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &Tensor,
    start_positions: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    full_attn_layer_idx: usize,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
    profile_context: Option<(usize, usize)>,
    cached_meta: Option<&CachedPagedDecodeMeta>,
) -> Result<Tensor> {
    let (batch, seq_len, _hidden) = x.dims3()?;
    let profile_device = x.device();
    anyhow::ensure!(batch > 0, "batched paged decode requires a non-empty batch");
    anyhow::ensure!(
        seq_len == 1,
        "batched contiguous paged attention requires one decode token per row"
    );
    anyhow::ensure!(
        block_tables.len() == batch && start_positions.len() == batch,
        "batched contiguous paged attention metadata length mismatch"
    );
    anyhow::ensure!(
        positions.elem_count() == 1 || positions.elem_count() == batch,
        "batched contiguous paged attention positions tensor must hold either a shared scalar or one entry per row"
    );
    anyhow::ensure!(
        !paged_cache.is_fp8(),
        "batched contiguous paged attention does not support FP8 caches"
    );
    anyhow::ensure!(
        full_attn_layer_idx < paged_cache.num_layers(),
        "batched contiguous paged attention layer index out of range"
    );

    // Phase 12-B-prime: drop the uniform-start_pos assertion stack. Per-row
    // K/V lengths are encoded via `seqused_k`, and per-row start positions
    // (used by RoPE + paged-KV slot indexing) are passed through as-is.
    //
    // When `cached_meta` is provided (set by the top-level batched decode
    // entry point), we skip the per-layer rebuild of the seqused_k /
    // block_table tensors. The cache is invariant across the 8 full-attn
    // layers within a step, so building once saves 7 HtoD launches per
    // step at bs > 1.
    let (
        max_seqlen_k,
        uniform_start_pos,
        max_blocks_per_seq,
        own_block_table_tensor,
        own_seqused_k_tensor,
        own_strict_start_slots,
    ): (usize, bool, usize, Option<Tensor>, Option<Tensor>, Option<Vec<u32>>) = match cached_meta
    {
        Some(meta) => (
            meta.max_seqlen_k,
            meta.uniform_start_pos,
            meta.max_blocks_per_seq,
            None,
            None,
            None,
        ),
        None => {
            let max_start_pos = *start_positions
                .iter()
                .max()
                .context("batched paged decode requires non-empty start_positions")?;
            let min_start_pos = *start_positions
                .iter()
                .min()
                .context("batched paged decode requires non-empty start_positions")?;
            let uniform_start_pos = max_start_pos == min_start_pos;
            let max_seqlen_k = max_start_pos + 1;

            // Build varlen metadata: per-row seqused_k tensor and a padded
            // [batch, max_blocks_per_seq] block_table tensor that indexes the
            // paged KV pool. `flash_attn_paged_decode_dyn_seqlen` masks padding
            // beyond each row's seqused_k.
            let page_block_size = paged_cache.block_size();
            let max_blocks_per_seq =
                ((max_seqlen_k + page_block_size - 1) / page_block_size).max(1);
            let mut block_table_vec = Vec::<u32>::with_capacity(batch * max_blocks_per_seq);
            let mut seqused_k_vec = Vec::<i32>::with_capacity(batch);
            for (row_idx, bt) in block_tables.iter().enumerate() {
                let row_seqlen = start_positions[row_idx] + 1;
                seqused_k_vec.push(
                    i32::try_from(row_seqlen)
                        .context("batched contiguous paged attention seqused_k exceeds i32 range")?,
                );
                let row_blocks = bt.blocks.as_slice();
                anyhow::ensure!(
                    row_blocks.len() * page_block_size >= row_seqlen,
                    "batched contiguous paged attention row {row_idx}: block_table covers {} tokens but row needs {}",
                    row_blocks.len() * page_block_size,
                    row_seqlen,
                );
                let pad_block = *row_blocks.last().unwrap_or(&0);
                for slot in 0..max_blocks_per_seq {
                    let phys = if slot < row_blocks.len() {
                        row_blocks[slot]
                    } else {
                        pad_block
                    };
                    block_table_vec.push(phys);
                }
            }

            // Strict-path slot_run vector kept as a fallback for when the
            // dyn_seqlen backend declines (e.g. kill switch armed). Only valid
            // when the live window is uniform across rows.
            let strict_start_slots: Option<Vec<u32>> = if uniform_start_pos {
                let live_window_starts = vec![0usize; batch];
                match paged_cache.contiguous_slot_run_starts(
                    block_tables,
                    &live_window_starts,
                    max_seqlen_k,
                ) {
                    Some(slots) => {
                        let v: Result<Vec<u32>> = slots
                            .iter()
                            .map(|&slot| {
                                u32::try_from(slot).context(
                                    "batched contiguous paged attention start slot exceeds u32 range",
                                )
                            })
                            .collect();
                        Some(v?)
                    }
                    None => None,
                }
            } else {
                None
            };

            let block_table_tensor = Tensor::from_slice(
                block_table_vec.as_slice(),
                (batch, max_blocks_per_seq),
                x.device(),
            )?
            .contiguous()?;
            let seqused_k_tensor =
                Tensor::from_slice(seqused_k_vec.as_slice(), batch, x.device())?.contiguous()?;

            (
                max_seqlen_k,
                uniform_start_pos,
                max_blocks_per_seq,
                Some(block_table_tensor),
                Some(seqused_k_tensor),
                strict_start_slots,
            )
        }
    };
    let _ = max_blocks_per_seq; // shape already baked into the cached tensor

    let use_metal_decode_gemv = start_positions.iter().all(|&p| p > 0)
        && !crate::mtp_debug::is_mtp_single_token_self_attn_armed();

    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let (q_raw, k, v) = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        kiln_nvtx::range!(c"kiln/proj/qkv_batch_decode");
        let out = full_attn_qkv_proj_decode_if(
            backend,
            use_metal_decode_gemv,
            x,
            attn_weights,
            lora_layer,
            lora_scale,
        )?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "qkv_proj_batch",
            seq_len,
            stage_profile,
        )?;
        out
    };

    let (q, gate) = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let out = if attn_output_gate {
            let q_raw = q_raw.reshape(((), seq_len, num_heads, head_dim * 2))?;
            let q = q_raw.narrow(3, 0, head_dim)?;
            let gate = q_raw.narrow(3, head_dim, head_dim)?;
            let gate = gate
                .contiguous()?
                .reshape(((), seq_len, num_heads * head_dim))?;
            (q.contiguous()?, Some(gate))
        } else {
            (q_raw.reshape(((), seq_len, num_heads, head_dim))?, None)
        };
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "qkv_split_batch",
            seq_len,
            stage_profile,
        )?;
        out
    };
    let k = k.reshape(((), seq_len, num_kv_heads, head_dim))?;
    let v = v.reshape(((), seq_len, num_kv_heads, head_dim))?;

    let (q, k) = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
        let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "qk_norm_batch",
            seq_len,
            stage_profile,
        )?;
        (q, k)
    };
    let (q, k) = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let out = if positions.elem_count() == 1 {
            // Shared scalar position: reuse the existing seq_len-major rope
            // path. cos/sin shape [1, half_rotary] broadcasts cleanly across
            // [batch, 1, num_heads, half_rotary].
            rotary_embedding_from_tensor(&q, &k, positions, head_dim, rotary_dim, inv_freq)?
        } else {
            // Per-row positions: swap batch <-> seq_len so cos/sin built from
            // [batch, half_rotary] aligns with the second dim of the q/k
            // tensors expected by `apply_rope`. After RoPE swap back.
            let q_swap = q.transpose(0, 1)?.contiguous()?;
            let k_swap = k.transpose(0, 1)?.contiguous()?;
            let (q_rot, k_rot) = rotary_embedding_from_tensor(
                &q_swap, &k_swap, positions, head_dim, rotary_dim, inv_freq,
            )?;
            (
                q_rot.transpose(0, 1)?.contiguous()?,
                k_rot.transpose(0, 1)?.contiguous()?,
            )
        };
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "rope_batch",
            seq_len,
            stage_profile,
        )?;
        out
    };
    // Q stays in [batch, 1, num_heads, head_dim] for the dyn_seqlen path; the
    // strict fallback below transposes lazily into the head-major layout it
    // requires.

    {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        if !paged_cache.write_token_major_native_batch(
            full_attn_layer_idx,
            block_tables,
            start_positions,
            &k,
            &v,
        )? {
            anyhow::bail!("batched contiguous paged attention KV write declined");
        }
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "kv_write_batch",
            seq_len,
            stage_profile,
        )?;
    }

    let (k_pool, v_pool) = paged_cache
        .pool_tensors(full_attn_layer_idx)
        .context("batched contiguous paged attention layer index out of range")?;
    // Prefer the once-per-step cached tensors when the caller built them;
    // otherwise use the per-layer ones we built above.
    let block_table_tensor: &Tensor = match (cached_meta, own_block_table_tensor.as_ref()) {
        (Some(meta), _) => &meta.block_table_tensor,
        (None, Some(t)) => t,
        (None, None) => unreachable!("cached_meta=None branch must build the block_table tensor"),
    };
    let seqused_k_tensor: &Tensor = match (cached_meta, own_seqused_k_tensor.as_ref()) {
        (Some(meta), _) => &meta.seqused_k_tensor,
        (None, Some(t)) => t,
        (None, None) => unreachable!("cached_meta=None branch must build the seqused_k tensor"),
    };
    let strict_start_slots: Option<&[u32]> = match (cached_meta, own_strict_start_slots.as_ref()) {
        (Some(meta), _) => meta.strict_start_slots.as_deref(),
        (None, Some(v)) => Some(v.as_slice()),
        (None, None) => None,
    };
    let page_block_size = paged_cache.block_size();
    let softmax_scale = 1.0f32 / (head_dim as f32).sqrt();
    let attn_output = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        // Phase 12-B-prime perf gate: dyn_seqlen handles divergent per-row
        // start_pos correctly but regressed synthetic c=8 throughput by ~61%
        // versus the post-#996 strict-path baseline under uniform load (which
        // is the common synthetic + most-production case). Route through the
        // strict head-major path when start_pos is uniform across the batch
        // (the pre-12-B-prime working path) and only fall through to
        // dyn_seqlen when rows actually diverge.
        //
        // Env switches:
        //   KILN_DISABLE_FUSED_PAGED_DECODE_DYN_SEQLEN_BATCH=1
        //     Force strict path everywhere (debug). Will fail loudly if a
        //     batch arrives with divergent start_pos because the strict
        //     kernel cannot handle that shape.
        //   KILN_FORCE_FUSED_PAGED_DECODE_DYN_SEQLEN_BATCH=1
        //     Force dyn_seqlen everywhere (A/B). Useful to reproduce the
        //     pre-fix throughput number or to validate dyn_seqlen
        //     correctness under uniform load.
        // KILN_DISABLE_* takes precedence over KILN_FORCE_* if both set.
        let kill_dyn_seqlen =
            std::env::var("KILN_DISABLE_FUSED_PAGED_DECODE_DYN_SEQLEN_BATCH").is_ok();
        let force_dyn_seqlen = !kill_dyn_seqlen
            && std::env::var("KILN_FORCE_FUSED_PAGED_DECODE_DYN_SEQLEN_BATCH").is_ok();
        let prefer_strict = !force_dyn_seqlen && uniform_start_pos && strict_start_slots.is_some();

        let try_strict = |out_acc: &mut Option<Tensor>| -> Result<()> {
            // Strict contiguous-batch path: pre-12-B-prime code path that
            // delivered PR #996's +10.76% c=8 throughput win. Requires
            // uniform start_pos + contiguous live KV. The strict kernel
            // expects head-major [batch, num_heads, 1, head_dim].
            let strict_slots = strict_start_slots.context(
                "batched contiguous paged attention requires uniform start_pos for the strict path",
            )?;
            let start_slots =
                Tensor::from_slice(strict_slots, batch, x.device())?.contiguous()?;
            let q_strict = {
                let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
                let q_strict = q.transpose(1, 2)?.contiguous()?;
                finish_full_attn_stage_profile(
                    profile_device,
                    profile_context,
                    "q_transpose_batch",
                    seq_len,
                    stage_profile,
                )?;
                q_strict
            };
            *out_acc = backend.flash_attn_paged_decode_contiguous_batch(
                &q_strict,
                k_pool,
                v_pool,
                &start_slots,
                max_seqlen_k,
                softmax_scale,
            )?;
            Ok(())
        };

        let try_dyn_seqlen = |out_acc: &mut Option<Tensor>| -> Result<()> {
            *out_acc = backend.flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
                &q,
                k_pool,
                v_pool,
                block_table_tensor,
                seqused_k_tensor,
                max_seqlen_k,
                page_block_size,
                softmax_scale,
                true,
            )?;
            Ok(())
        };

        let mut out: Option<Tensor> = None;
        if kill_dyn_seqlen {
            // Strict-only; non-uniform batches will surface the strict_slots
            // context error.
            try_strict(&mut out)?;
        } else if prefer_strict {
            try_strict(&mut out)?;
            if out.is_none() {
                // Strict backend declined (e.g. Metal CPU fallback path).
                // Fall through to dyn_seqlen.
                try_dyn_seqlen(&mut out)?;
            }
        } else {
            try_dyn_seqlen(&mut out)?;
            if out.is_none() && uniform_start_pos && strict_start_slots.is_some() {
                // dyn_seqlen backend declined; the strict path can still
                // serve uniform batches. Divergent batches have no fallback
                // and will surface as the final context error below.
                try_strict(&mut out)?;
            }
        }
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "decode_attn_contiguous_batch",
            seq_len,
            stage_profile,
        )?;
        out.context("backend declined batched contiguous paged attention")?
    };

    // Both kernels feed o_proj a row-major [batch, 1, num_heads * head_dim].
    // The Metal strict kernel already returns that 3-D shape; the dyn_seqlen
    // kernel returns 4-D [batch, 1, num_heads, head_dim], so flatten the trailing
    // axes here. The reshape is a no-op for the 3-D case.
    let attn_output = if attn_output.dims().len() == 4 {
        attn_output.reshape((batch, seq_len, num_heads * head_dim))?
    } else {
        attn_output
    };

    let attn_output = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let out =
            attention_output_gate_decode_if(use_metal_decode_gemv, attn_output, gate.as_ref())?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "attn_gate_batch",
            seq_len,
            stage_profile,
        )?;
        out
    };
    let out = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        kiln_nvtx::range!(c"kiln/proj/o_batch_decode");
        let out = linear_with_lora_t_backend_decode_if(
            Some(backend),
            use_metal_decode_gemv,
            &attn_output,
            &attn_weights.o_proj_t,
            lora_layer.and_then(|l| l.o_proj.as_ref()),
            lora_scale,
        )?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "o_proj_batch",
            seq_len,
            stage_profile,
        )?;
        out
    };
    Ok(out)
}

/// Grouped-query attention using a paged KV cache.
///
/// Same computation as [`gqa_attention`] but reads/writes K/V through a
/// [`PagedKvCache`] and [`BlockTable`] instead of a contiguous [`KvCache`].
/// This enables multiple concurrent sequences to share a fixed KV cache pool.
///
/// The caller must ensure the block table has enough blocks allocated for all
/// positions up to `positions.last() + 1`.
pub fn gqa_attention_paged(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &Tensor,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    gqa_attention_paged_with_rope_tables(
        backend,
        x,
        attn_weights,
        positions,
        start_pos,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        inv_freq,
        None,
        rms_norm_eps,
        paged_cache,
        block_table,
        full_attn_layer_idx,
        attn_output_gate,
        lora,
        #[cfg(feature = "cuda")]
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn gqa_attention_paged_with_rope_tables(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &Tensor,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rope_tables: Option<(&Tensor, &Tensor)>,
    rms_norm_eps: f64,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
    #[cfg(feature = "cuda")] graph_inputs: Option<&PagedDecodeGraphInputs<'_>>,
) -> Result<Tensor> {
    let (_batch, seq_len, _hidden) = x.dims3()?;
    let profile_device = x.device();
    let profile_context =
        profile_full_attn_stages_enabled().then_some((full_attn_layer_idx, start_pos));
    let subop_armed = crate::mtp_debug::is_subop_capture_armed();
    let b12_layer_31 = crate::mtp_debug::current_b12_layer_is_31();
    let use_metal_decode_gemv =
        seq_len == 1 && start_pos > 0 && !crate::mtp_debug::is_mtp_single_token_self_attn_armed();

    // Project to Q, K, V (with optional LoRA delta and output gate split)
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let (q_raw, k_raw, v) = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        kiln_nvtx::range!(c"kiln/proj/qkv");
        let out = full_attn_qkv_proj_decode_if(
            backend,
            use_metal_decode_gemv,
            x,
            attn_weights,
            lora_layer,
            lora_scale,
        )?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "qkv_proj",
            seq_len,
            stage_profile,
        )?;
        out
    };
    // Phase B7b sub-op taps: post-projection (pre-split). `q_raw` may include
    // the gate half when `attn_output_gate` is on, so its trailing dim is 2H.
    if subop_armed {
        let _ = crate::mtp_debug::capture_subop("post_q_proj_raw", &q_raw);
        let _ = crate::mtp_debug::capture_subop("post_k_proj", &k_raw);
        let _ = crate::mtp_debug::capture_subop("post_v_proj", &v);
    }
    // Phase B9 H3 alias: pre_gated_attn_split is the q_raw tensor before the
    // (q, gate) narrow split. Captured as alias of post_q_proj_raw so the
    // comparator can locate H3 zone divergence by name.
    if subop_armed {
        let _ = crate::mtp_debug::capture_subop("pre_gated_attn_split", &q_raw);
    }
    // Phase B12 layer-31 GQA taps: q_proj / k_proj / v_proj. These are
    // the post-projection tensors before the gate split. No-op unless
    // layer 31 is executing with B12 capture armed.
    if b12_layer_31 {
        crate::mtp_debug::capture_b12_gqa_tap("q_proj", &q_raw)?;
        crate::mtp_debug::capture_b12_gqa_tap("k_proj", &k_raw)?;
        crate::mtp_debug::capture_b12_gqa_tap("v_proj", &v)?;
    }

    let fused_qkv_prep: Option<(Tensor, Tensor, Option<Tensor>)> = {
        #[cfg(feature = "cuda")]
        {
            if seq_len == 1
                && !cuda_fused_attn_decode_qkv_prep_disabled()
                && !subop_armed
                && !b12_layer_31
                && !any_tensor_tracks_op(&[
                    &q_raw,
                    &k_raw,
                    &attn_weights.q_norm,
                    &attn_weights.k_norm,
                ])
            {
                if let Some((cos, sin)) = rope_tables {
                    if kiln_rmsnorm_kernel::supports_attn_decode_qkv_prep(
                        &q_raw,
                        &k_raw,
                        &attn_weights.q_norm,
                        &attn_weights.k_norm,
                        cos,
                        sin,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        rotary_dim,
                        attn_output_gate,
                    ) {
                        let stage_profile =
                            start_full_attn_stage_profile(profile_device, profile_context)?;
                        kiln_nvtx::range!(c"kiln/attn/qkv_prep_cuda_fused");
                        let out = kiln_rmsnorm_kernel::fused_attn_decode_qkv_prep(
                            &q_raw,
                            &k_raw,
                            &attn_weights.q_norm,
                            &attn_weights.k_norm,
                            cos,
                            sin,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                            rotary_dim,
                            attn_output_gate,
                            rms_norm_eps as f32,
                        )
                        .context("cuda fused attn decode qkv prep failed")?;
                        finish_full_attn_stage_profile(
                            profile_device,
                            profile_context,
                            "qkv_split_qk_norm_rope",
                            seq_len,
                            stage_profile,
                        )?;
                        Some(out)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    };

    let (q, k, gate) = if let Some((q, k, gate)) = fused_qkv_prep {
        (q, k, gate)
    } else {
        let (q, gate) = {
            let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
            kiln_nvtx::range!(c"kiln/proj/qkv_split");
            let out = if attn_output_gate {
                let q_raw = q_raw.reshape(((), seq_len, num_heads, head_dim * 2))?;
                let q = q_raw.narrow(3, 0, head_dim)?;
                let gate = q_raw.narrow(3, head_dim, head_dim)?;
                let gate = gate
                    .contiguous()?
                    .reshape(((), seq_len, num_heads * head_dim))?;
                (q.contiguous()?, Some(gate))
            } else {
                let q = q_raw.reshape(((), seq_len, num_heads, head_dim))?;
                (q, None)
            };
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "qkv_split",
                seq_len,
                stage_profile,
            )?;
            out
        };
        // After the gate split, q is the rotation target.
        if subop_armed {
            let _ = crate::mtp_debug::capture_subop("post_q_split", &q);
        }
        // Phase B9 H3 alias: post_gated_attn_split_value mirrors post_q_split.
        if subop_armed {
            let _ = crate::mtp_debug::capture_subop("post_gated_attn_split_value", &q);
            if let Some(ref g) = gate {
                let _ = crate::mtp_debug::capture_subop("post_gate_split", g);
                // Phase B9 H3 alias: post_gated_attn_split_gate mirrors post_gate_split.
                let _ = crate::mtp_debug::capture_subop("post_gated_attn_split_gate", g);
            }
        }

        let k = k_raw.reshape(((), seq_len, num_kv_heads, head_dim))?;

        // Phase B9 H2 taps: pre_qk_norm_{q,k} are the per-head reshaped tensors
        // immediately before per-head RMSNorm. pre_qk_norm_q is alias of
        // post_q_split; pre_qk_norm_k is genuinely new (post_k_proj is pre-reshape).
        if subop_armed {
            let _ = crate::mtp_debug::capture_subop("pre_qk_norm_q", &q);
            let _ = crate::mtp_debug::capture_subop("pre_qk_norm_k", &k);
        }

        // QK-norm
        let (q, k) = {
            let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
            kiln_nvtx::range!(c"kiln/attn/qk_norm");
            let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
            let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;
            let out = (q, k);
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "qk_norm",
                seq_len,
                stage_profile,
            )?;
            out
        };
        if subop_armed {
            let _ = crate::mtp_debug::capture_subop("post_q_norm", &q);
            let _ = crate::mtp_debug::capture_subop("post_k_norm", &k);
        }
        // Phase B9 H2 aliases: post_qk_norm_{q,k} mirror post_{q,k}_norm.
        if subop_armed {
            let _ = crate::mtp_debug::capture_subop("post_qk_norm_q", &q);
            let _ = crate::mtp_debug::capture_subop("post_qk_norm_k", &k);
        }
        // Phase B12 layer-31 GQA taps: qk_norm_q / qk_norm_k. Post per-head
        // RMSNorm, pre-RoPE. Shape [B, T, num_heads, head_dim] /
        // [B, T, num_kv_heads, head_dim].
        if b12_layer_31 {
            crate::mtp_debug::capture_b12_gqa_tap("qk_norm_q", &q)?;
            crate::mtp_debug::capture_b12_gqa_tap("qk_norm_k", &k)?;
        }

        // RoPE — only rotate first rotary_dim dimensions
        // Use the GPU tensor variant so positions remain at a stable GPU address
        // (critical for CUDA graph replay correctness)
        let (q, k) = {
            let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
            kiln_nvtx::range!(c"kiln/attn/rope");
            let out = if let Some((cos, sin)) = rope_tables {
                rotary_embedding_from_tables(&q, &k, cos, sin, head_dim, rotary_dim)?
            } else {
                rotary_embedding_from_tensor(&q, &k, positions, head_dim, rotary_dim, inv_freq)?
            };
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "rope",
                seq_len,
                stage_profile,
            )?;
            out
        };
        if subop_armed {
            let _ = crate::mtp_debug::capture_subop("post_q_rope", &q);
            let _ = crate::mtp_debug::capture_subop("post_k_rope", &k);
        }
        // Phase B12 layer-31 GQA taps: rope_q / rope_k. Post-RoPE, pre-transpose.
        // These are intermediates that HF can only expose via a forward hook on
        // the attention module's q_proj/k_proj output + manual re-run of the
        // rotary function in the comparator — the Python dump script emits a
        // NOTE rather than failing when these HF taps are absent.
        if b12_layer_31 {
            crate::mtp_debug::capture_b12_gqa_tap("rope_q", &q)?;
            crate::mtp_debug::capture_b12_gqa_tap("rope_k", &k)?;
        }

        (q, k, gate)
    };

    let v = v.reshape(((), seq_len, num_kv_heads, head_dim))?;

    // Keep the cache-native token-major K/V views for paged writes. Attention
    // still wants head-major tensors, but the cache pool stores
    // `[slot, kv_head, dim]`, so using these avoids a transpose back during
    // prefill.
    let k_cache_token_major = k.clone();
    let v_cache_token_major = v.clone();

    // Transpose Q to [batch, heads, seq_len, head_dim]. K/V are transposed
    // lazily only on paths that consume the current tile directly; later
    // prefill tiles and speculative verifier windows read full head-major K/V
    // back from the paged cache instead.
    let q = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        kiln_nvtx::range!(c"kiln/attn/qkv_transpose");
        let q = q.transpose(1, 2)?.contiguous()?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "q_transpose",
            seq_len,
            stage_profile,
        )?;
        q
    };

    let total_seq_len = start_pos + seq_len;

    // Initial prefill fast path: when there is no prefix history yet
    // (`start_pos == 0`), the current K/V tensors already cover the entire
    // attention window. Route prefill through the backend flash-attn path
    // directly and only write K/V into the paged cache once for future decode.
    // This avoids a pointless write-then-read round-trip through
    // `PagedKvCache` on the first prompt tile.
    if seq_len > 1
        && start_pos == 0
        && (backend.supports_flash_attn_prefill_head_major()
            || backend.supports_flash_attn_prefill())
    {
        kiln_nvtx::range!(c"kiln/attn/full/prefill_initial");
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let k_head = k_cache_token_major.transpose(1, 2)?.contiguous()?;
        let v_head = v_cache_token_major.transpose(1, 2)?.contiguous()?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "prefill_kv_head_layout",
            seq_len,
            stage_profile,
        )?;
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let attn_output = if let Some(attn_output) =
            flash_attention_forward_head_major(backend, &q, &k_head, &v_head, num_heads, head_dim)?
        {
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "prefill_attn_head_major",
                seq_len,
                stage_profile,
            )?;
            Some(attn_output)
        } else if backend.supports_flash_attn_prefill() {
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "prefill_attn_head_major",
                seq_len,
                stage_profile,
            )?;
            let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
            let q_prefill = q.transpose(1, 2)?.contiguous()?; // -> [batch, seq_len, num_heads, head_dim]
            let k_prefill = k_cache_token_major.contiguous()?; // [batch, seq_len, num_kv_heads, head_dim]
            let v_prefill = v_cache_token_major.contiguous()?; // [batch, seq_len, num_kv_heads, head_dim]
            let out = flash_attention_forward(
                backend,
                &q_prefill,
                &k_prefill,
                &v_prefill,
                num_heads,
                num_kv_heads,
                head_dim,
            )?;
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "prefill_attn_fallback",
                seq_len,
                stage_profile,
            )?;
            out
        } else {
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "prefill_attn_head_major",
                seq_len,
                stage_profile,
            )?;
            None
        };

        if let Some(attn_output) = attn_output {
            let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
            {
                kiln_nvtx::range!(c"kiln/kv/copy");
                if !paged_cache.write_token_major_native(
                    full_attn_layer_idx,
                    block_table,
                    start_pos,
                    &k_cache_token_major,
                    &v_cache_token_major,
                )? {
                    paged_cache
                        .write(
                            full_attn_layer_idx,
                            block_table,
                            start_pos,
                            &k_head,
                            &v_head,
                        )
                        .context("paged KV cache write failed")?;
                }
            }
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "kv_write",
                seq_len,
                stage_profile,
            )?;

            // Phase B12 layer-31 GQA tap: attn_out. Captured AFTER the gate
            // multiply (if `attn_output_gate`) and BEFORE o_proj, so it
            // matches the HF reference's `attn_output = ... * sigmoid_gate`
            // tap point. Shape: [B, T, num_heads * head_dim].
            let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
            let attn_output = attention_output_gate_decode_if(false, attn_output, gate.as_ref())?;
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "attn_gate",
                seq_len,
                stage_profile,
            )?;
            if b12_layer_31 {
                crate::mtp_debug::capture_b12_gqa_tap("attn_out", &attn_output)?;
            }
            let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
            let out = {
                kiln_nvtx::range!(c"kiln/proj/o");
                linear_with_lora_t_backend_decode_if(
                    Some(backend),
                    false,
                    &attn_output,
                    &attn_weights.o_proj_t,
                    lora_layer.and_then(|l| l.o_proj.as_ref()),
                    lora_scale,
                )?
            };
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "o_proj",
                seq_len,
                stage_profile,
            )?;
            // Phase B12 layer-31 GQA tap: o_proj output (post-o_proj).
            if b12_layer_31 {
                crate::mtp_debug::capture_b12_gqa_tap("o_proj", &out)?;
            }
            return Ok(out);
        }
    }

    // Phase C8: when the MTP forward step has armed single-token
    // self-attention, the MTP layer attends only to the just-computed K/V
    // (kv_len = 1, no history). Skip the paged-cache write/read and the
    // fused paged-decode kernel entirely so the per-step (k, v) above
    // becomes the SDPA input. Cleared back to `false` by the matching
    // `disarm_mtp_single_token_self_attn` in `mtp_forward_step`, so non-MTP
    // attention calls on this thread are unaffected.
    let single_token_self_attn = crate::mtp_debug::is_mtp_single_token_self_attn_armed();

    // Write new K/V into paged cache.
    if !single_token_self_attn {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        kiln_nvtx::range!(c"kiln/kv/copy");
        let graph_write_done = {
            #[cfg(feature = "cuda")]
            {
                if let Some(inputs) = graph_inputs {
                    paged_cache.write_token_major_native_graph_slot(
                        full_attn_layer_idx,
                        &k_cache_token_major,
                        &v_cache_token_major,
                        inputs.kv_slot,
                    )?
                } else {
                    false
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        if !graph_write_done
            && !paged_cache.write_token_major_native(
                full_attn_layer_idx,
                block_table,
                start_pos,
                &k_cache_token_major,
                &v_cache_token_major,
            )?
        {
            let k_head = k_cache_token_major.transpose(1, 2)?.contiguous()?;
            let v_head = v_cache_token_major.transpose(1, 2)?.contiguous()?;
            paged_cache
                .write(
                    full_attn_layer_idx,
                    block_table,
                    start_pos,
                    &k_head,
                    &v_head,
                )
                .context("paged KV cache write failed")?;
        }
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "kv_write",
            seq_len,
            stage_profile,
        )?;
    }

    // Fast path: fused paged-decode flash-attention kernel.
    // Eliminates the materializing `paged_cache.read()` (an `index_select` /
    // u8→bf16 dequant) on the decode hot path. Limited to:
    //   * Backends that advertise `supports_flash_attn_paged_decode()`
    //     (CUDA + bf16 today)
    //   * Decode steps (seq_len == 1)
    //   * Non-FP8 caches (the kernel reads bf16 pool slots directly)
    //   * Page sizes that divide kBlockN=128 (block_size=16 satisfies this)
    //   * Single sequence with physically contiguous block allocation
    //     (kiln's BlockManager allocates blocks in order from a free list, so
    //     a freshly-allocated single sequence is always contiguous)
    //   * Phase C8: not in single-token self-attn mode (kernel reads the
    //     full cache history, defeating the kv_len = 1 contract).
    if seq_len == 1
        && !single_token_self_attn
        && !paged_cache.is_fp8()
        && (num_heads / num_kv_heads) > 1
        && !fused_paged_decode_disabled()
        && backend.supports_flash_attn_paged_decode()
        && !crate::mtp_debug::is_c7_sdpa_capture_armed()
    {
        // Open the fused-decode range around the call so the kernel work is
        // attributed to it. When the eligibility checks inside reject (return
        // None) the range still closes here and the fallback range below
        // takes over for the rest of the iteration. Eligibility-rejection is
        // cheap so the over-attribution is small.
        let out_opt = {
            kiln_nvtx::range!(c"kiln/attn/full/decode_fused");
            try_flash_attn_paged_decode(
                backend,
                &q,
                paged_cache,
                block_table,
                full_attn_layer_idx,
                total_seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                gate.as_ref(),
                use_metal_decode_gemv,
                attn_weights,
                lora_layer,
                lora_scale,
                #[cfg(feature = "cuda")]
                graph_inputs,
                profile_context,
            )?
        };
        if let Some(out) = out_opt {
            return Ok(out);
        }
    }

    // Open the fallback-decode range BEFORE the paged_cache.read so the read's
    // gather/dequant ucopy is attributed to it. The range stays open through
    // the GQA decode work below; it harmlessly also covers the prefill FA-2
    // path (which has its own inner range and returns from inside it). The
    // range is bound to the function scope so it always closes on return.
    let _decode_fallback_nvtx = if seq_len == 1 {
        Some(kiln_nvtx::Range::push(c"kiln/attn/full/decode_fallback"))
    } else {
        None
    };

    // Read full K/V from paged cache (all positions 0..start_pos+seq_len).
    // Phase C8: when single_token_self_attn is armed (MTP inner GQA call),
    // attend only to the just-computed (k, v) — kv_len = 1, no cache read.
    // This matches the Qwen3-Next MTP reference contract where the inner
    // block performs single-token self-attention without a growing KV history.
    let (k, v, kv_len) = if single_token_self_attn {
        (
            k_cache_token_major.transpose(1, 2)?.contiguous()?,
            v_cache_token_major.transpose(1, 2)?.contiguous()?,
            1usize,
        )
    } else {
        let prefix_only_prefill = seq_len > 1
            && start_pos > 0
            && !paged_cache.is_fp8()
            && backend.supports_flash_attn_prefill_head_major()
            && !crate::mtp_debug::is_c7_sdpa_capture_armed();
        let prefix_append_fast = if prefix_only_prefill
            && start_pos >= PAGED_KV_HEAD_MAJOR_READ_MIN_TOKENS
            && backend.supports_paged_kv_head_major_read_append_token_major()
        {
            contiguous_slot_run_start(block_table, paged_cache.block_size(), 0, start_pos)
                .and_then(|start_slot| {
                    paged_cache
                        .pool_tensors(full_attn_layer_idx)
                        .map(|(k_pool, v_pool)| (start_slot, k_pool, v_pool))
                })
                .map(|(start_slot, k_pool, v_pool)| {
                    backend.paged_kv_head_major_read_append_token_major(
                        k_pool,
                        v_pool,
                        start_slot,
                        start_pos,
                        &k_cache_token_major,
                        &v_cache_token_major,
                    )
                })
                .transpose()?
                .flatten()
        } else {
            None
        };
        let fast_read_len = if prefix_only_prefill {
            start_pos
        } else {
            total_seq_len
        };
        let fast_read = if seq_len > 1
            && fast_read_len >= PAGED_KV_HEAD_MAJOR_READ_MIN_TOKENS
            && !paged_cache.is_fp8()
            && backend.supports_paged_kv_head_major_read()
            && backend.supports_flash_attn_prefill_head_major()
        {
            contiguous_slot_run_start(block_table, paged_cache.block_size(), 0, fast_read_len)
                .and_then(|start_slot| {
                    paged_cache
                        .pool_tensors(full_attn_layer_idx)
                        .map(|(k_pool, v_pool)| (start_slot, k_pool, v_pool))
                })
                .map(|(start_slot, k_pool, v_pool)| {
                    backend.paged_kv_head_major_read(k_pool, v_pool, start_slot, fast_read_len)
                })
                .transpose()?
                .flatten()
        } else {
            None
        };
        let (k, v) = if prefix_only_prefill {
            match prefix_append_fast {
                Some((k, v)) => (k, v),
                None => {
                    let (prefix_k, prefix_v) = match fast_read {
                        Some((k, v)) => (k, v),
                        None => paged_cache
                            .read(full_attn_layer_idx, block_table, start_pos)
                            .context("paged KV cache prefix read failed")?,
                    };
                    let current_k = k_cache_token_major.transpose(1, 2)?.contiguous()?;
                    let current_v = v_cache_token_major.transpose(1, 2)?.contiguous()?;
                    (
                        Tensor::cat(&[&prefix_k, &current_k], 2)?,
                        Tensor::cat(&[&prefix_v, &current_v], 2)?,
                    )
                }
            }
        } else {
            match fast_read {
                Some((k, v)) => (k, v),
                None => {
                    let stage_profile =
                        start_full_attn_stage_profile(profile_device, profile_context)?;
                    let out = paged_cache
                        .read(full_attn_layer_idx, block_table, total_seq_len)
                        .context("paged KV cache read failed")?;
                    finish_full_attn_stage_profile(
                        profile_device,
                        profile_context,
                        "kv_read",
                        seq_len,
                        stage_profile,
                    )?;
                    out
                }
            }
        };
        (k, v, total_seq_len)
    };

    // Multi-token append / speculative verify with prefix history. `read`
    // already returns head-major K/V; on Metal, keep Q/K/V in that layout and
    // avoid token-major transposes plus GQA K/V expansion.
    if seq_len > 1
        && backend.supports_flash_attn_prefill_head_major()
        && !crate::mtp_debug::is_c7_sdpa_capture_armed()
    {
        kiln_nvtx::range!(c"kiln/attn/full/prefill_head_major");
        if let Some(attn_output) =
            flash_attention_forward_head_major(backend, &q, &k, &v, num_heads, head_dim)?
        {
            let attn_output = attention_output_gate_decode_if(false, attn_output, gate.as_ref())?;
            if b12_layer_31 {
                crate::mtp_debug::capture_b12_gqa_tap("attn_out", &attn_output)?;
            }
            let out = {
                kiln_nvtx::range!(c"kiln/proj/o");
                linear_with_lora_t_backend_decode_if(
                    Some(backend),
                    false,
                    &attn_output,
                    &attn_weights.o_proj_t,
                    lora_layer.and_then(|l| l.o_proj.as_ref()),
                    lora_scale,
                )?
            };
            if b12_layer_31 {
                crate::mtp_debug::capture_b12_gqa_tap("o_proj", &out)?;
            }
            return Ok(out);
        }
    }

    // Fused-attention path for prefill with existing prefix history
    // (`start_pos > 0`). Initial prefill is special-cased above so we do not
    // materialize the same K/V we just produced.
    // Paged cache returns [batch, heads, kv_len, head_dim] — transpose to
    // [batch, kv_len, heads, head_dim] for the backend kernel.
    if seq_len > 1 && backend.supports_flash_attn_prefill() {
        kiln_nvtx::range!(c"kiln/attn/full/prefill");
        let q = q.transpose(1, 2)?.contiguous()?; // -> [batch, seq_len, num_heads, head_dim]
        let k = k.transpose(1, 2)?.contiguous()?; // -> [batch, kv_len, num_kv_heads, head_dim]
        let v = v.transpose(1, 2)?.contiguous()?; // -> [batch, kv_len, num_kv_heads, head_dim]
        if let Some(attn_output) =
            flash_attention_forward(backend, &q, &k, &v, num_heads, num_kv_heads, head_dim)?
        {
            let attn_output = attention_output_gate_decode_if(false, attn_output, gate.as_ref())?;
            // Phase B12 layer-31 GQA tap (secondary prefill path).
            if b12_layer_31 {
                crate::mtp_debug::capture_b12_gqa_tap("attn_out", &attn_output)?;
            }
            let out = {
                kiln_nvtx::range!(c"kiln/proj/o");
                linear_with_lora_t_backend_decode_if(
                    Some(backend),
                    false,
                    &attn_output,
                    &attn_weights.o_proj_t,
                    lora_layer.and_then(|l| l.o_proj.as_ref()),
                    lora_scale,
                )?
            };
            if b12_layer_31 {
                crate::mtp_debug::capture_b12_gqa_tap("o_proj", &out)?;
            }
            return Ok(out);
        }
    }

    // GQA head expansion and attention
    let gqa_ratio = num_heads / num_kv_heads;
    let batch = k.dim(0)?;

    // Optimized decode path (seq_len == 1): reshape Q instead of expanding K/V.
    // Q is [batch, num_heads, 1, head_dim] (1 token) while K/V is
    // [batch, num_kv_heads, kv_len, head_dim] (full history). Expanding K/V
    // copies kv_len * head_dim * num_kv_heads data gqa_ratio times.
    // Instead, group Q heads to match KV heads and compute per-group attention.
    if seq_len == 1 && gqa_ratio > 1 {
        let scale = (head_dim as f64).sqrt();

        // Phase C7 SDPA bisect: capture pre-SDPA Q/K/V and causal-mask taps
        // BEFORE the grouping reshape, in the canonical HF shapes:
        //   Q: [batch, num_heads, q_len=1, head_dim]
        //   K: [batch, num_kv_heads, kv_len, head_dim] (unexpanded — HF
        //      reference dumps the same pre-repeat_kv form)
        //   V: same shape as K
        //   causal_mask: scalar 0 placeholder (decode has q_len=1 and attends
        //      to all kv_len positions, so no mask is applied)
        let c7_armed = crate::mtp_debug::is_c7_sdpa_capture_armed();
        if c7_armed {
            crate::mtp_debug::capture_c7_sdpa_tap("pre_sdpa_q", &q)?;
            crate::mtp_debug::capture_c7_sdpa_tap("pre_sdpa_k", &k)?;
            crate::mtp_debug::capture_c7_sdpa_tap("pre_sdpa_v", &v)?;
            let empty_mask = candle_core::Tensor::zeros((), candle_core::DType::F32, q.device())?;
            crate::mtp_debug::capture_c7_sdpa_tap("causal_mask", &empty_mask)?;
        }

        // Reshape Q: [batch, num_heads, 1, head_dim]
        //          -> [batch, num_kv_heads, gqa_ratio, 1, head_dim]
        //          -> [batch * num_kv_heads, gqa_ratio, 1, head_dim]
        // K:         [batch, num_kv_heads, kv_len, head_dim]
        //          -> [batch * num_kv_heads, kv_len, head_dim]
        // V:         same as K
        let (q_grouped, k_flat, v_flat) = {
            let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
            let q_grouped = q
                .reshape((batch, num_kv_heads, gqa_ratio, 1, head_dim))?
                .reshape((batch * num_kv_heads, gqa_ratio, 1, head_dim))?
                .contiguous()?;
            // Unsqueeze K/V to [batch*num_kv_heads, 1, kv_len, head_dim] so that
            // broadcast_matmul pairs each Q group with its own KV head (dim 0),
            // broadcasting over the gqa_ratio dim (dim 1).  Without the unsqueeze
            // the 3-D K would be padded to [1, batch*num_kv_heads, ...] and the
            // gqa_ratio dim would incorrectly index into different KV heads.
            let k_flat = k
                .reshape((batch * num_kv_heads, kv_len, head_dim))?
                .unsqueeze(1)?
                .contiguous()?;
            let v_flat = v
                .reshape((batch * num_kv_heads, kv_len, head_dim))?
                .unsqueeze(1)?
                .contiguous()?;
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "decode_group_layout",
                seq_len,
                stage_profile,
            )?;
            (q_grouped, k_flat, v_flat)
        };

        // Attention scores: [batch*num_kv_heads, gqa_ratio, 1, kv_len]
        let attn_scores = {
            let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
            let attn_scores = q_grouped.broadcast_matmul(&k_flat.transpose(2, 3)?.contiguous()?)?;
            let attn_scores = (attn_scores / scale)?;
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "decode_scores",
                seq_len,
                stage_profile,
            )?;
            attn_scores
        };

        // Phase C7: reshape grouped scores back to canonical
        // [batch, num_heads, 1, kv_len] for diff against HF.
        if c7_armed {
            let scores_canonical = attn_scores
                .reshape((batch, num_kv_heads, gqa_ratio, 1, kv_len))?
                .reshape((batch, num_heads, 1, kv_len))?;
            crate::mtp_debug::capture_c7_sdpa_tap("attn_scores_pre_softmax", &scores_canonical)?;
        }

        // No causal mask needed for decode (q_len=1 attends to everything)
        let attn_weights_softmax = {
            let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
            let out = cuda_softmax_last_dim(&attn_scores)?;
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "decode_softmax",
                seq_len,
                stage_profile,
            )?;
            out
        };

        // Phase C7: reshape grouped probs back to canonical
        // [batch, num_heads, 1, kv_len] for diff against HF.
        if c7_armed {
            let probs_canonical = attn_weights_softmax
                .reshape((batch, num_kv_heads, gqa_ratio, 1, kv_len))?
                .reshape((batch, num_heads, 1, kv_len))?;
            crate::mtp_debug::capture_c7_sdpa_tap("attn_probs", &probs_canonical)?;
        }

        // Weighted sum: [batch*num_kv_heads, gqa_ratio, 1, head_dim]
        let attn_output = {
            let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
            let attn_output = attn_weights_softmax.broadcast_matmul(&v_flat)?;

            // Reshape back: -> [batch, num_kv_heads * gqa_ratio, 1, head_dim]
            //               == [batch, num_heads, 1, head_dim]
            let attn_output = attn_output
                .reshape((batch, num_heads, 1, head_dim))?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, 1, num_heads * head_dim))?;
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "decode_weighted_sum",
                seq_len,
                stage_profile,
            )?;
            attn_output
        };
        if subop_armed {
            let _ = crate::mtp_debug::capture_subop("post_attn_raw", &attn_output);
        }

        // Phase C7: final SDPA output tap at the same point as post_attn_raw,
        // shape [batch, q_len=1, num_heads*head_dim] = [1, 1, 4096].
        if c7_armed {
            crate::mtp_debug::capture_c7_sdpa_tap("attn_out", &attn_output)?;
        }

        let attn_output = {
            let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
            let out =
                attention_output_gate_decode_if(use_metal_decode_gemv, attn_output, gate.as_ref())?;
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "attn_gate",
                seq_len,
                stage_profile,
            )?;
            out
        };
        if subop_armed {
            let _ = crate::mtp_debug::capture_subop("post_attn_gated", &attn_output);
        }
        // Phase B12 layer-31 GQA tap (grouped decode path).
        if b12_layer_31 {
            crate::mtp_debug::capture_b12_gqa_tap("attn_out", &attn_output)?;
        }
        let out = {
            let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
            kiln_nvtx::range!(c"kiln/proj/o");
            let out = linear_with_lora_t_backend_decode_if(
                Some(backend),
                use_metal_decode_gemv,
                &attn_output,
                &attn_weights.o_proj_t,
                lora_layer.and_then(|l| l.o_proj.as_ref()),
                lora_scale,
            )?;
            finish_full_attn_stage_profile(
                profile_device,
                profile_context,
                "o_proj",
                seq_len,
                stage_profile,
            )?;
            out
        };
        if subop_armed {
            let _ = crate::mtp_debug::capture_subop("post_o_proj", &out);
        }
        if b12_layer_31 {
            crate::mtp_debug::capture_b12_gqa_tap("o_proj", &out)?;
        }
        return Ok(out);
    }

    // Standard path (prefill without flash-attn, or gqa_ratio == 1)
    let (k, v) = if gqa_ratio > 1 {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let k = k
            .unsqueeze(2)?
            .expand(&[batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        let v = v
            .unsqueeze(2)?
            .expand(&[batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "prefill_gqa_expand",
            seq_len,
            stage_profile,
        )?;
        (k, v)
    } else {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let out = (k.contiguous()?, v.contiguous()?);
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "prefill_kv_contiguous",
            seq_len,
            stage_profile,
        )?;
        out
    };

    // Scaled dot-product attention
    let scale = (head_dim as f64).sqrt();
    let attn_scores = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let attn_scores = q.broadcast_matmul(&k.t()?)?;
        let attn_scores = (attn_scores / scale)?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "prefill_scores",
            seq_len,
            stage_profile,
        )?;
        attn_scores
    };

    let past_len = kv_len - seq_len;
    let attn_scores = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let attn_scores = apply_causal_mask_with_offset(&attn_scores, seq_len, kv_len, past_len)?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "prefill_mask",
            seq_len,
            stage_profile,
        )?;
        attn_scores
    };

    let attn_weights_softmax = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let out = cuda_softmax_last_dim(&attn_scores)?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "prefill_softmax",
            seq_len,
            stage_profile,
        )?;
        out
    };
    let attn_output = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let out = attn_weights_softmax.broadcast_matmul(&v)?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "prefill_weighted_sum",
            seq_len,
            stage_profile,
        )?;
        out
    };

    // Transpose back and output projection
    let attn_output = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let out = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            (),
            seq_len,
            num_heads * head_dim,
        ))?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "prefill_output_layout",
            seq_len,
            stage_profile,
        )?;
        out
    };
    if subop_armed {
        let _ = crate::mtp_debug::capture_subop("post_attn_raw", &attn_output);
    }

    let attn_output = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        let out =
            attention_output_gate_decode_if(use_metal_decode_gemv, attn_output, gate.as_ref())?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "attn_gate",
            seq_len,
            stage_profile,
        )?;
        out
    };
    if subop_armed {
        let _ = crate::mtp_debug::capture_subop("post_attn_gated", &attn_output);
    }
    // Phase B12 layer-31 GQA tap (standard fallback path).
    if b12_layer_31 {
        crate::mtp_debug::capture_b12_gqa_tap("attn_out", &attn_output)?;
    }

    let out = {
        let stage_profile = start_full_attn_stage_profile(profile_device, profile_context)?;
        kiln_nvtx::range!(c"kiln/proj/o");
        let out = linear_with_lora_t_backend_decode_if(
            Some(backend),
            use_metal_decode_gemv,
            &attn_output,
            &attn_weights.o_proj_t,
            lora_layer.and_then(|l| l.o_proj.as_ref()),
            lora_scale,
        )?;
        finish_full_attn_stage_profile(
            profile_device,
            profile_context,
            "o_proj",
            seq_len,
            stage_profile,
        )?;
        out
    };
    if subop_armed {
        let _ = crate::mtp_debug::capture_subop("post_o_proj", &out);
    }
    if b12_layer_31 {
        crate::mtp_debug::capture_b12_gqa_tap("o_proj", &out)?;
    }
    Ok(out)
}

/// Apply a causal (lower-triangular) mask to attention scores.
/// Sets future positions to -inf so softmax zeroes them out.
#[allow(dead_code)]
fn apply_causal_mask(scores: &Tensor, seq_len: usize) -> Result<Tensor> {
    apply_causal_mask_with_offset(scores, seq_len, seq_len, 0)
}

/// Apply a causal mask with support for KV cache offset.
///
/// When using a KV cache, Q has `q_len` new positions and K/V has `kv_len` total
/// positions (past_len cached + q_len new). Each query position `i` (representing
/// absolute position `past_len + i`) can attend to all KV positions up to and
/// including itself: positions `0..past_len + i + 1`.
///
/// `scores`: [batch, heads, q_len, kv_len]
/// `q_len`: number of new query positions
/// `kv_len`: total KV length (past_len + q_len)
/// `past_len`: number of cached positions before the new tokens
fn apply_causal_mask_with_offset(
    scores: &Tensor,
    q_len: usize,
    kv_len: usize,
    past_len: usize,
) -> Result<Tensor> {
    if q_len <= 1 && kv_len <= 1 {
        return Ok(scores.clone());
    }
    // During decode (q_len=1), the single new token can attend to all kv_len
    // positions (all past + itself), so no masking needed.
    if q_len == 1 {
        return Ok(scores.clone());
    }
    let device = scores.device();
    // Build a [q_len, kv_len] mask: 0 for allowed, -inf for masked
    // Query position i (absolute: past_len + i) can attend to KV positions 0..past_len+i+1
    let mask: Vec<f32> = (0..q_len)
        .flat_map(|i| {
            let max_kv = past_len + i + 1; // last allowed KV position (exclusive)
            (0..kv_len).map(move |j| if j < max_kv { 0.0 } else { f32::NEG_INFINITY })
        })
        .collect();
    let mask = Tensor::new(mask, device)?.reshape((1, 1, q_len, kv_len))?;
    let mask = mask.to_dtype(scores.dtype())?;
    let out = scores.broadcast_add(&mask)?;
    Ok(out)
}

/// Single transformer block: norm -> attention -> residual -> norm -> FFN -> residual.
///
/// `x`: [batch, seq_len, hidden_size]
/// `layer`: weights for this transformer layer
/// `positions`: position indices for RoPE (absolute positions)
/// `num_heads`: number of query attention heads
/// `num_kv_heads`: number of key/value attention heads
/// `head_dim`: dimension per head
/// `rotary_dim`: number of head dims to rotate (partial RoPE)
/// `inv_freq`: cached RoPE frequency table (built once via [`compute_rotary_inv_freq`])
/// `rms_norm_eps`: epsilon for RMSNorm
/// `kv_cache`: optional KV cache for incremental decoding
/// `full_attn_layer_idx`: index into the KV cache's layer array
///
/// Returns: [batch, seq_len, hidden_size]
pub fn transformer_block(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &[u32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    kv_cache: Option<&mut KvCache>,
    full_attn_layer_idx: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let attn_weights = match &layer.attention {
        GpuAttentionWeights::Full(w) => w,
        GpuAttentionWeights::Linear(_) => {
            anyhow::bail!("transformer_block only supports full attention layers (not linear/GDN)")
        }
    };
    let (_batch, seq_len, _hidden) = x.dims3()?;
    let use_metal_decode_ffn = seq_len == 1
        && kv_cache.is_some()
        && !crate::mtp_debug::is_mtp_single_token_self_attn_armed();

    if let Some(out) = transformer_block_detached_cuda_prefill_chunked(
        backend,
        x,
        layer,
        config,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        inv_freq,
        rms_norm_eps,
        kv_cache.is_some(),
        full_attn_layer_idx,
        lora,
    )? {
        return Ok(out);
    }

    // Pre-attention norm
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_attn");
        rms_norm(x, &layer.input_layernorm, rms_norm_eps)?
    };

    // Self-attention
    let attn_out = gqa_attention(
        backend,
        &normed,
        attn_weights,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        inv_freq,
        rms_norm_eps,
        kv_cache,
        full_attn_layer_idx,
        config.attn_output_gate,
        lora,
    )?;

    // Residual connection
    let x = {
        kiln_nvtx::range!(c"kiln/residual");
        (x + attn_out)?
    };

    // Post-attention norm
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_mlp");
        rms_norm(&x, &layer.post_attention_layernorm, rms_norm_eps)?
    };

    // Feed-forward network
    let ffn_out = if use_metal_decode_ffn {
        swiglu_ffn_metal_decode(&normed, &layer.mlp, lora)?
    } else {
        swiglu_ffn(&normed, &layer.mlp, lora)?
    };

    // Residual connection
    let out = {
        kiln_nvtx::range!(c"kiln/residual");
        (x + ffn_out)?
    };
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn transformer_block_detached_cuda_prefill_chunked(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &[u32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    has_kv_cache: bool,
    full_attn_layer_idx: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Option<Tensor>> {
    if backend.name() != "cuda" || has_kv_cache || x.track_op() {
        return Ok(None);
    }
    let (_batch, seq_len, _hidden) = x.dims3()?;
    if !streaming_prefill_enabled_for(x.device(), seq_len) {
        return Ok(None);
    }
    let tile_size = streaming_tile_tokens_for(x.device());
    if tile_size == 0 || tile_size >= seq_len {
        return Ok(None);
    }

    let GpuAttentionWeights::Full(attn_weights) = &layer.attention else {
        return Ok(None);
    };

    tracing::info!(
        layer = full_attn_layer_idx,
        seq_len,
        tile_size,
        "detached CUDA full-attention prefill chunked"
    );

    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_attn_chunked");
        rms_norm(x, &layer.input_layernorm, rms_norm_eps)?
    };
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let k = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        &normed,
        &attn_weights.k_proj_t,
        lora_layer.and_then(|l| l.k_proj.as_ref()),
        lora_scale,
    )
    .context("chunked full-attention k projection")?
    .reshape(((), seq_len, num_kv_heads, head_dim))
    .context("chunked full-attention k reshape")?;
    let v = linear_with_lora_t_backend_decode_if(
        Some(backend),
        false,
        &normed,
        &attn_weights.v_proj_t,
        lora_layer.and_then(|l| l.v_proj.as_ref()),
        lora_scale,
    )
    .context("chunked full-attention v projection")?
    .reshape(((), seq_len, num_kv_heads, head_dim))
    .context("chunked full-attention v reshape")?;
    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)
        .context("chunked full-attention k norm")?;
    let (k, _) = rotary_embedding(&k, &k, positions, head_dim, rotary_dim, inv_freq)
        .context("chunked full-attention k rotary")?;

    let mut output_tiles = Vec::with_capacity(seq_len.div_ceil(tile_size));
    let mut tile_start = 0usize;
    while tile_start < seq_len {
        let tile_len = (seq_len - tile_start).min(tile_size);
        let tile_end = tile_start + tile_len;

        let normed_tile = normed.narrow(1, tile_start, tile_len).with_context(|| {
            format!("chunked full-attention normed tile [{tile_start}, {tile_end})")
        })?;
        let q_raw = q_proj_forward_decode_if(
            Some(backend),
            false,
            &normed_tile,
            attn_weights,
            lora_layer.and_then(|l| l.q_proj.as_ref()),
            lora_scale,
        )
        .with_context(|| {
            format!("chunked full-attention q projection [{tile_start}, {tile_end})")
        })?;
        let (q_tile, gate_tile) = if config.attn_output_gate {
            let q_raw = q_raw
                .reshape(((), tile_len, num_heads, head_dim * 2))
                .with_context(|| {
                    format!("chunked full-attention q/gate reshape [{tile_start}, {tile_end})")
                })?;
            let q = q_raw
                .narrow(3, 0, head_dim)
                .with_context(|| {
                    format!("chunked full-attention q split [{tile_start}, {tile_end})")
                })?
                .contiguous()
                .context("chunked full-attention q contiguous")?;
            let gate = q_raw
                .narrow(3, head_dim, head_dim)
                .with_context(|| {
                    format!("chunked full-attention gate split [{tile_start}, {tile_end})")
                })?
                .contiguous()
                .context("chunked full-attention gate contiguous")?
                .reshape(((), tile_len, num_heads * head_dim))
                .context("chunked full-attention gate reshape")?;
            (q, Some(gate))
        } else {
            (
                q_raw
                    .reshape(((), tile_len, num_heads, head_dim))
                    .with_context(|| {
                        format!("chunked full-attention q reshape [{tile_start}, {tile_end})")
                    })?,
                None,
            )
        };
        let q_tile = rms_norm(&q_tile, &attn_weights.q_norm, rms_norm_eps)
            .context("chunked full-attention q norm")?;
        let tile_positions = &positions[tile_start..tile_end];
        let (q_tile, _) = rotary_embedding(
            &q_tile,
            &q_tile,
            tile_positions,
            head_dim,
            rotary_dim,
            inv_freq,
        )
        .with_context(|| format!("chunked full-attention q rotary [{tile_start}, {tile_end})"))?;
        let k_prefix = k.narrow(1, 0, tile_end).with_context(|| {
            format!("chunked full-attention k prefix [0, {tile_end}) for tile {tile_start}")
        })?;
        let v_prefix = v.narrow(1, 0, tile_end).with_context(|| {
            format!("chunked full-attention v prefix [0, {tile_end}) for tile {tile_start}")
        })?;
        let tile_prepared = GqaAttentionPrepared {
            q: q_tile,
            k: k_prefix,
            v: v_prefix,
            gate: None,
        };

        let attn_core =
            gqa_attention_core_prefill(backend, &tile_prepared, num_heads, num_kv_heads, head_dim)
                .with_context(|| {
                    format!("chunked full-attention core tile [{tile_start}, {tile_end})")
                })?;
        let attn_output = gqa_attention_apply_output_gate(attn_core, gate_tile.as_ref())
            .with_context(|| {
                format!("chunked full-attention gate tile [{tile_start}, {tile_end})")
            })?;
        let attn_out =
            gqa_attention_output_projection(backend, &attn_output, attn_weights, false, lora)
                .with_context(|| {
                    format!("chunked full-attention o-proj tile [{tile_start}, {tile_end})")
                })?;
        let x_tile = x.narrow(1, tile_start, tile_len).with_context(|| {
            format!("chunked full-attention residual tile [{tile_start}, {tile_end})")
        })?;
        let residual = (&x_tile + attn_out).with_context(|| {
            format!("chunked full-attention attention residual tile [{tile_start}, {tile_end})")
        })?;
        let normed_post = rms_norm(&residual, &layer.post_attention_layernorm, rms_norm_eps)
            .with_context(|| {
                format!("chunked full-attention post norm tile [{tile_start}, {tile_end})")
            })?;
        let ffn_out = swiglu_ffn(&normed_post, &layer.mlp, lora).with_context(|| {
            format!("chunked full-attention MLP tile [{tile_start}, {tile_end})")
        })?;
        let out_tile = (residual + ffn_out)
            .with_context(|| {
                format!("chunked full-attention output tile [{tile_start}, {tile_end})")
            })?
            .detach();
        output_tiles.push(out_tile);

        tile_start = tile_end;
    }

    let output_refs: Vec<&Tensor> = output_tiles.iter().collect();
    let output = Tensor::cat(&output_refs, 1)
        .context("chunked full-attention output cat")?
        .detach();
    Ok(Some(output))
}

/// Transformer block using paged KV cache.
///
/// Same as [`transformer_block`] but reads/writes K/V through a [`PagedKvCache`]
/// and [`BlockTable`] instead of a contiguous [`KvCache`].
pub fn transformer_block_paged(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &Tensor,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rms_norm_eps: f64,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    transformer_block_paged_with_rope_tables(
        backend,
        x,
        layer,
        config,
        positions,
        start_pos,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        inv_freq,
        None,
        rms_norm_eps,
        paged_cache,
        block_table,
        full_attn_layer_idx,
        lora,
        #[cfg(feature = "cuda")]
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn transformer_block_paged_with_rope_tables(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &Tensor,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &Tensor,
    rope_tables: Option<(&Tensor, &Tensor)>,
    rms_norm_eps: f64,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
    #[cfg(feature = "cuda")] graph_inputs: Option<&PagedDecodeGraphInputs<'_>>,
    profile_mlp_context: Option<(usize, usize)>,
) -> Result<Tensor> {
    let attn_weights = match &layer.attention {
        GpuAttentionWeights::Full(w) => w,
        GpuAttentionWeights::Linear(_) => {
            anyhow::bail!(
                "transformer_block_paged only supports full attention layers (not linear/GDN)"
            )
        }
    };
    let (_batch, seq_len, _hidden) = x.dims3()?;
    let subop_armed = crate::mtp_debug::is_subop_capture_armed();
    let b12_layer_31 = crate::mtp_debug::current_b12_layer_is_31();
    let use_metal_decode_ffn = seq_len == 1
        && start_pos > 0
        && !b12_layer_31
        && !crate::mtp_debug::is_mtp_single_token_self_attn_armed();

    // Pre-attention norm
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_attn");
        rms_norm(x, &layer.input_layernorm, rms_norm_eps)?
    };
    if subop_armed {
        let _ = crate::mtp_debug::capture_subop("post_pre_attn_norm", &normed);
    }
    // Phase B12: layer-31 GQA sub-op tap #1. Named `post_input_norm` to
    // match the HF reference-dump naming. No-op unless we are on base-model
    // layer 31 with the B12 capture window armed.
    if b12_layer_31 {
        crate::mtp_debug::capture_b12_gqa_tap("post_input_norm", &normed)?;
    }

    // Self-attention with paged cache
    let attn_out = gqa_attention_paged_with_rope_tables(
        backend,
        &normed,
        attn_weights,
        positions,
        start_pos,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        inv_freq,
        rope_tables,
        rms_norm_eps,
        paged_cache,
        block_table,
        full_attn_layer_idx,
        config.attn_output_gate,
        lora,
        #[cfg(feature = "cuda")]
        graph_inputs,
    )?;
    if subop_armed {
        let _ = crate::mtp_debug::capture_subop("post_attn_block", &attn_out);
    }

    // Residual connection
    let x = {
        kiln_nvtx::range!(c"kiln/residual");
        (x + attn_out)?
    };
    if subop_armed {
        let _ = crate::mtp_debug::capture_subop("post_attn_residual", &x);
    }

    // Post-attention norm
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_mlp");
        rms_norm(&x, &layer.post_attention_layernorm, rms_norm_eps)?
    };
    if subop_armed {
        let _ = crate::mtp_debug::capture_subop("post_pre_mlp_norm", &normed);
    }
    // Phase B12: layer-31 GQA sub-op tap — post_attn_norm. Named to match
    // the HF reference. No-op unless layer 31 + armed.
    if b12_layer_31 {
        crate::mtp_debug::capture_b12_gqa_tap("post_attn_norm", &normed)?;
    }

    // Feed-forward network. For layer 31 with B12 armed, route through a
    // sub-op-tapping path that exposes mlp_gate / mlp_up / mlp_down; the
    // standard `swiglu_ffn` fuses those and is fine for everyone else.
    let ffn_out = if b12_layer_31 {
        swiglu_ffn_b12_tapped(&normed, &layer.mlp, lora)?
    } else if use_metal_decode_ffn {
        swiglu_ffn_backend_profiled(
            backend,
            &normed,
            &layer.mlp,
            lora,
            true,
            profile_mlp_context,
        )?
    } else {
        swiglu_ffn_backend_profiled(
            backend,
            &normed,
            &layer.mlp,
            lora,
            false,
            profile_mlp_context,
        )?
    };
    if subop_armed {
        let _ = crate::mtp_debug::capture_subop("post_mlp", &ffn_out);
    }

    // Residual connection
    let out = {
        kiln_nvtx::range!(c"kiln/residual");
        (x + ffn_out)?
    };
    // Note: the final block output (`out`) is dumped as `post_layer` at the
    // outer MTP call site, so we do not re-capture it here.
    Ok(out)
}

/// Batched decode variant of [`transformer_block_paged`] for full-attention
/// layers whose paged-KV windows are contiguous and share one decode position.
///
/// This wraps [`gqa_attention_paged_decode_contiguous_batch`] with the block's
/// pre-attention norm, residuals, post-attention norm, and MLP. Linear/GDN
/// layers are intentionally out of scope; they use `LinearAttentionState`
/// batching instead.
#[allow(clippy::too_many_arguments)]
pub fn transformer_block_paged_decode_contiguous_batch(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &Tensor,
    start_positions: &[usize],
    inv_freq: &Tensor,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    full_attn_layer_idx: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
    full_attn_profile_context: Option<(usize, usize)>,
    mlp_profile_context: Option<(usize, usize)>,
    cached_meta: Option<&CachedPagedDecodeMeta>,
) -> Result<Tensor> {
    let attn_weights = match &layer.attention {
        GpuAttentionWeights::Full(w) => w,
        GpuAttentionWeights::Linear(_) => {
            anyhow::bail!(
                "batched contiguous paged transformer decode only supports full attention layers"
            )
        }
    };
    let (_batch, seq_len, _hidden) = x.dims3()?;
    anyhow::ensure!(
        seq_len == 1,
        "batched contiguous paged transformer decode requires one token per row"
    );
    anyhow::ensure!(
        !start_positions.is_empty(),
        "batched contiguous paged transformer decode requires a non-empty batch"
    );
    // Phase 12-B-prime: per-row start positions are allowed; the SwiGLU MLP
    // decode-gemv hint must hold for every row, so require all > 0.
    let use_metal_decode_ffn = start_positions.iter().all(|&p| p > 0)
        && !crate::mtp_debug::is_mtp_single_token_self_attn_armed();

    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_attn_batch_decode");
        rms_norm(x, &layer.input_layernorm, config.rms_norm_eps)?
    };
    let attn_out = gqa_attention_paged_decode_contiguous_batch(
        backend,
        &normed,
        attn_weights,
        positions,
        start_positions,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.rotary_dim(),
        inv_freq,
        config.rms_norm_eps,
        paged_cache,
        block_tables,
        full_attn_layer_idx,
        config.attn_output_gate,
        lora,
        full_attn_profile_context,
        cached_meta,
    )?;
    let x = {
        kiln_nvtx::range!(c"kiln/residual_batch_decode");
        (x + attn_out)?
    };
    let normed = {
        kiln_nvtx::range!(c"kiln/norm/pre_mlp_batch_decode");
        rms_norm(&x, &layer.post_attention_layernorm, config.rms_norm_eps)?
    };
    let ffn_out = swiglu_ffn_backend_profiled(
        backend,
        &normed,
        &layer.mlp,
        lora,
        use_metal_decode_ffn,
        mlp_profile_context,
    )?;
    let out = {
        kiln_nvtx::range!(c"kiln/residual_batch_decode");
        (x + ffn_out)?
    };
    Ok(out)
}

/// Strict batched single-token paged decode up through the final transformer
/// block.
///
/// This is the model-loop counterpart to
/// [`transformer_block_paged_decode_contiguous_batch`]. It accepts one token
/// per batch row, a block table per row, a common decode position, and an
/// optional batch-shaped [`LinearAttentionState`]. It returns final hidden
/// states with shape `[batch, 1, hidden_size]`.
///
/// The helper is deliberately narrower than the general scheduler contract:
/// every row must share the same `start_pos`, full-attention rows must satisfy
/// the contiguous paged-KV constraints enforced by E340/E341, and LoRA/debug
/// capture paths remain owned by the rowwise entry points until scheduler
/// integration needs them.
#[allow(clippy::too_many_arguments)]
fn model_forward_paged_decode_contiguous_batch_hidden(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    model_forward_paged_decode_contiguous_batch_hidden_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_tables,
        start_positions,
        linear_state.as_deref_mut(),
        lora,
        None,
        None,
    )
}

/// Implementation backing `model_forward_paged_decode_contiguous_batch_hidden`
/// plus the upcoming batched CUDA graph wrapper. When
/// `stable_positions_gpu` / `stable_token_ids_gpu` are `Some`, the
/// function skips the per-step host→device builds for those tensors
/// and reads from the caller-owned device pointers instead — exactly
/// the invariant CUDA graph capture/replay needs.
#[allow(clippy::too_many_arguments)]
fn model_forward_paged_decode_contiguous_batch_hidden_inner(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    stable_positions_gpu: Option<&Tensor>,
    stable_token_ids_gpu: Option<&Tensor>,
) -> Result<Tensor> {
    let batch = token_ids.len();
    anyhow::ensure!(batch > 0, "batched paged decode requires a non-empty batch");
    anyhow::ensure!(
        block_tables.len() == batch && start_positions.len() == batch,
        "batched paged decode metadata length mismatch"
    );
    let max_start_pos = *start_positions
        .iter()
        .max()
        .context("batched paged decode requires a non-empty start_positions")?;

    if weights
        .layers
        .iter()
        .any(|layer| matches!(layer.attention, GpuAttentionWeights::Linear(_)))
    {
        let state_batch = linear_state
            .as_ref()
            .context("batched paged decode requires LinearAttentionState for GDN layers")?
            .batch_size()?;
        anyhow::ensure!(
            state_batch == batch,
            "batched paged decode LinearAttentionState batch mismatch ({state_batch} vs {batch})"
        );
    }

    let device = weights.embed_tokens.device();
    // Embedding lookup. When the caller supplies a stable `[batch] u32`
    // token-id tensor on the device (CUDA graph capture path), use it
    // via the index-based lookup — the device pointer stays valid
    // across replays. Otherwise build fresh from the host slice.
    let mut hidden = if let Some(token_ids_gpu) = stable_token_ids_gpu {
        embedding_lookup_from_weights_with_index(token_ids_gpu, weights)?.unsqueeze(1)?
    } else {
        embedding_lookup_from_weights(token_ids, weights)?.unsqueeze(1)?
    };
    // When every row decodes at the same position (the common case — all
    // requests admitted with same-length prompts or all admitted at the same
    // decode step), pass a single-element positions tensor so the full-attn
    // RoPE picks the fast scalar-broadcast path (`positions.elem_count() == 1`)
    // and skips the 4-transpose+contig dance the per-row path needs to align
    // cos/sin with the batch dim. nsys at bs=16 (post-broadcast-matmul fix)
    // showed ~32 RoPE transpose+contig copies per decode step routing through
    // copy2d_bf16; this elides them when positions happen to be uniform.
    //
    // CUDA graph capture path: when `stable_positions_gpu` is `Some`,
    // skip both branches above and use the caller-owned device buffer
    // so the captured RoPE kernels read from a graph-stable pointer.
    // The bench shows the per-step `Tensor::from_slice` here is a tiny
    // HtoD launch (one per step), so the win comes from graph
    // captureability, not from elimination of the copy itself.
    let first_pos = start_positions[0];
    let positions_uniform = start_positions.iter().all(|&p| p == first_pos);
    let positions_owned: Option<Tensor> = if stable_positions_gpu.is_none() {
        Some(if positions_uniform {
            Tensor::from_slice(&[first_pos as f32], 1usize, device)?
        } else {
            let positions_f32: Vec<f32> = start_positions.iter().map(|&p| p as f32).collect();
            Tensor::from_slice(positions_f32.as_slice(), batch, device)?
        })
    } else {
        None
    };
    let positions: &Tensor = stable_positions_gpu
        .unwrap_or_else(|| positions_owned.as_ref().expect("positions_owned built above when stable was None"));
    let use_metal_decode_ffn = start_positions.iter().all(|&p| p > 0)
        && !crate::mtp_debug::is_mtp_single_token_self_attn_armed();
    let profile_full_attn_stages = profile_full_attn_stages_enabled();
    let profile_gdn_stages = profile_gdn_stages_enabled();
    let profile_mlp_stages = profile_mlp_stages_enabled();

    // Build the per-step paged-decode metadata once when there are any
    // full-attention layers in the model. Each gqa call within this step
    // would otherwise rebuild the seqused_k + padded block_table tensors
    // (one HtoD launch each) per layer (8× on Qwen3.5-4B); hoisting saves
    // 14 launches per step. Skip the build entirely on linear-only models.
    let has_full_attention_layer = weights
        .layers
        .iter()
        .any(|layer| matches!(layer.attention, GpuAttentionWeights::Full(_)));
    let cached_paged_meta: Option<CachedPagedDecodeMeta> = if has_full_attention_layer {
        Some(
            CachedPagedDecodeMeta::build(device, paged_cache, block_tables, start_positions)
                .context("build cached paged decode metadata for batched step")?,
        )
    } else {
        None
    };
    let cached_paged_meta_ref = cached_paged_meta.as_ref();

    let mut full_attn_idx = 0usize;
    let mut linear_attn_idx = 0usize;
    for (i, layer) in weights.layers.iter().enumerate() {
        let layer_lora: Option<(&LoraLayerWeights, f32)> =
            lora.and_then(|lw| lw.layers.get(i).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                hidden = transformer_block_paged_decode_contiguous_batch(
                    backend,
                    &hidden,
                    layer,
                    config,
                    positions,
                    start_positions,
                    &weights.rotary_inv_freq,
                    paged_cache,
                    block_tables,
                    full_attn_idx,
                    layer_lora,
                    profile_full_attn_stages.then_some((full_attn_idx, max_start_pos)),
                    profile_mlp_stages.then_some((i, max_start_pos)),
                    cached_paged_meta_ref,
                )
                .with_context(|| {
                    format!("batched transformer block {i} (full attention, paged)")
                })?;
                full_attn_idx += 1;
            }
            GpuAttentionWeights::Linear(lin_weights) => {
                let state = linear_state.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("batched linear attention state required for GDN layer {i}")
                })?;
                let normed = {
                    kiln_nvtx::range!(c"kiln/norm/pre_attn_batch_decode");
                    rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?
                };
                let attn_out = gated_deltanet_forward_decode_if(
                    backend,
                    &normed,
                    lin_weights,
                    config,
                    &mut state.recurrent_states[linear_attn_idx],
                    &mut state.conv_states[linear_attn_idx],
                    false,
                    false,
                    use_metal_decode_ffn,
                    use_metal_decode_ffn,
                    profile_gdn_stages.then_some((i, max_start_pos)),
                    true,
                    true,
                    layer_lora,
                )
                .with_context(|| {
                    format!("batched gated deltanet layer {i} (linear attention, paged)")
                })?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual_batch_decode");
                    (hidden + attn_out)?
                };
                let normed_post = {
                    kiln_nvtx::range!(c"kiln/norm/pre_mlp_batch_decode");
                    rms_norm(
                        &hidden,
                        &layer.post_attention_layernorm,
                        config.rms_norm_eps,
                    )?
                };
                let ffn_out = swiglu_ffn_backend_profiled(
                    backend,
                    &normed_post,
                    &layer.mlp,
                    layer_lora,
                    use_metal_decode_ffn,
                    profile_mlp_stages.then_some((i, max_start_pos)),
                )?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual_batch_decode");
                    (hidden + ffn_out)?
                };
                linear_attn_idx += 1;
            }
        }
    }

    Ok(hidden)
}

/// Strict batched single-token paged decode for model-forward integration.
///
/// This is the model-loop counterpart to
/// [`transformer_block_paged_decode_contiguous_batch`]. It accepts one token
/// per batch row, a block table per row, per-row decode positions, and an
/// optional batch-shaped [`LinearAttentionState`]. It returns full logits with
/// shape `[batch, 1, vocab_size]`.
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_decode_contiguous_batch(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    let hidden = model_forward_paged_decode_contiguous_batch_hidden(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_tables,
        start_positions,
        linear_state,
        lora,
    )?;
    let logits = {
        kiln_nvtx::range!(c"kiln/lm_head_batch_decode");
        let normed = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
        lm_head_forward_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)?
    };
    Ok(logits)
}

/// Strict batched single-token paged decode that returns greedy next-token IDs
/// without materializing full logits when a backend has a fused argmax path.
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_decode_contiguous_batch_greedy(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Vec<u32>> {
    let hidden = model_forward_paged_decode_contiguous_batch_hidden(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_tables,
        start_positions,
        linear_state,
        lora,
    )?;
    let token_ids = {
        kiln_nvtx::range!(c"kiln/lm_head_batch_argmax_decode");
        let normed = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
        lm_head_argmax_rows_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)?
    };
    Ok(token_ids)
}

/// Full model forward pass: embedding → N transformer blocks → final norm → LM head → logits.
///
/// `token_ids`: 1-D slice of token IDs for the input sequence.
/// `weights`: pre-loaded GPU tensors for all model parameters.
/// `config`: model architecture configuration.
/// `kv_cache`: optional KV cache for incremental decoding. When provided, `token_ids`
///   should contain only the new (not yet cached) tokens, and positions are computed
///   starting from `kv_cache.seq_len()`.
///
/// Returns logits tensor with shape [1, seq_len, vocab_size].
///
/// Notes:
/// - Qwen3.5-4B uses weight tying: the LM head reuses `embed_tokens` transposed.
/// - Linear attention (Gated DeltaNet) layers are not yet implemented and will
///   be skipped with an identity pass-through.
/// - After this function returns, the caller must call `kv_cache.advance(token_ids.len())`
///   to update the cached sequence length.
pub fn model_forward(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    mut kv_cache: Option<&mut KvCache>,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    let seq_len = token_ids.len();

    // 1. Embedding lookup: [seq_len, hidden_size]
    let mut hidden = embedding_lookup_from_weights(token_ids, weights)?;

    // Add batch dimension: [1, seq_len, hidden_size]
    hidden = hidden.unsqueeze(0)?;

    // Position indices for RoPE — absolute positions accounting for cached tokens
    let offset = kv_cache.as_ref().map_or(0, |c| c.seq_len());
    let positions: Vec<u32> = (offset..offset + seq_len).map(|p| p as u32).collect();
    let use_metal_decode_ffn =
        seq_len == 1 && offset > 0 && !crate::mtp_debug::is_mtp_single_token_self_attn_armed();

    // 2. Loop through all transformer layers
    // Track full-attention layer index (0-based counter of only full-attn layers)
    let mut full_attn_idx: usize = 0;
    let mut linear_attn_idx: usize = 0;
    for (i, layer) in weights.layers.iter().enumerate() {
        // Get LoRA weights for this layer, if available
        let layer_lora: Option<(&LoraLayerWeights, f32)> =
            lora.and_then(|lw| lw.layers.get(i).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                // Reborrow the cache for each layer call
                let cache_ref = kv_cache.as_mut().map(|c| &mut **c);
                hidden = transformer_block(
                    backend,
                    &hidden,
                    layer,
                    config,
                    &positions,
                    config.num_attention_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.rotary_dim(),
                    &weights.rotary_inv_freq,
                    config.rms_norm_eps,
                    cache_ref,
                    full_attn_idx,
                    layer_lora,
                )
                .with_context(|| format!("transformer block {i} (full attention)"))?;
                full_attn_idx += 1;
            }
            GpuAttentionWeights::Linear(lin_weights) => {
                let state = linear_state.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("linear attention state required for GDN layers (layer {i})")
                })?;
                // Pre-attention RMSNorm
                let normed = {
                    kiln_nvtx::range!(c"kiln/norm/pre_attn");
                    rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?
                };
                // Gated DeltaNet linear attention
                let attn_out = gated_deltanet_forward_decode_if(
                    backend,
                    &normed,
                    lin_weights,
                    config,
                    &mut state.recurrent_states[linear_attn_idx],
                    &mut state.conv_states[linear_attn_idx],
                    /* capture_b11_taps = */ false,
                    /* capture_c41_taps = */ false,
                    /* use_fused_gdn_gates = */ true,
                    use_metal_decode_ffn,
                    None,
                    true,
                    true,
                    layer_lora,
                )
                .with_context(|| format!("gated deltanet layer {i} (linear attention)"))?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    (hidden + attn_out)?
                };
                // Post-attention RMSNorm + FFN
                let normed_post = {
                    kiln_nvtx::range!(c"kiln/norm/pre_mlp");
                    rms_norm(
                        &hidden,
                        &layer.post_attention_layernorm,
                        config.rms_norm_eps,
                    )?
                };
                let ffn_out = if use_metal_decode_ffn {
                    swiglu_ffn_metal_decode(&normed_post, &layer.mlp, layer_lora)?
                } else {
                    swiglu_ffn(&normed_post, &layer.mlp, layer_lora)?
                };
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    (hidden + ffn_out)?
                };
                linear_attn_idx += 1;
            }
        }
    }

    // 3. Final RMSNorm + 4. LM head projection (weight-tied: embed_tokens^T)
    // hidden: [1, seq_len, hidden_size], embed_tokens: [vocab_size, hidden_size]
    // logits = hidden @ embed_tokens^T -> [1, seq_len, vocab_size]
    let logits = {
        kiln_nvtx::range!(c"kiln/lm_head");
        hidden = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
        lm_head_forward_backend_decode_if(Some(backend), &hidden, &weights.embed_tokens_t)?
    };

    Ok(logits)
}

/// Run a subset of transformer layers on an existing hidden state.
///
/// Processes layers `[start_layer..end_layer)` without embedding or LM head.
/// Used by gradient checkpointing to recompute individual segments.
///
/// `hidden`: [1, seq_len, hidden_size] — input hidden state.
/// `positions`: absolute position indices for RoPE.
/// `linear_state`: mutable linear attention state (only entries for layers in range are touched).
///
/// Returns: [1, seq_len, hidden_size] — output hidden state.
pub fn model_forward_segment(
    backend: &dyn BackendRuntime,
    mut hidden: Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &[u32],
    start_layer: usize,
    end_layer: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    // Count full-attention and linear-attention layers before start_layer
    // so we index into the right KV cache / linear state slots.
    let mut full_attn_idx: usize = (0..start_layer)
        .filter(|&i| matches!(&weights.layers[i].attention, GpuAttentionWeights::Full(_)))
        .count();
    let mut linear_attn_idx: usize = (0..start_layer)
        .filter(|&i| matches!(&weights.layers[i].attention, GpuAttentionWeights::Linear(_)))
        .count();

    // Phase 10: training-time streaming GDN prefill.
    //
    // When `KILN_STREAMING_PREFILL=1` and the segment's seq_len exceeds the
    // configured tile size, GDN layers run as a sequence of smaller tiles
    // threading `LinearAttentionState` per tile. Full-attention layers always
    // run monolithically — training has no KV cache to thread across tiles,
    // so a tiled full-attn layer would only attend within its tile and break
    // the global causal mask. Inter-layer hidden activations stay at full T
    // shape; only the per-call GDN intermediates (causal_conv1d F32
    // promotion, l2_normalize F32 buffers, chunkwise scratch) shrink to per-
    // tile shape. This is the peak transient allocation that pushes T=8192
    // SFT past the A6000 ceiling per the PR #634 audit.
    let (_, seq_len, _) = hidden.dims3()?;
    let stream_device = hidden.device().clone();
    let streaming = streaming_prefill_enabled_for(&stream_device, seq_len);
    let stream_tile = if streaming {
        streaming_tile_tokens_for(&stream_device)
    } else {
        0
    };
    let stream_active = streaming && stream_tile > 0 && seq_len > stream_tile;

    for i in start_layer..end_layer {
        let layer = &weights.layers[i];
        let layer_lora: Option<(&LoraLayerWeights, f32)> =
            lora.and_then(|lw| lw.layers.get(i).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                // Training doesn't use KV cache
                hidden = transformer_block(
                    backend,
                    &hidden,
                    layer,
                    config,
                    positions,
                    config.num_attention_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.rotary_dim(),
                    &weights.rotary_inv_freq,
                    config.rms_norm_eps,
                    None, // no KV cache for training
                    full_attn_idx,
                    layer_lora,
                )
                .with_context(|| format!("segment transformer block {i} (full attention)"))?;
                full_attn_idx += 1;
            }
            GpuAttentionWeights::Linear(lin_weights) => {
                let state = linear_state.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("linear attention state required for GDN layers (layer {i})")
                })?;
                let normed = {
                    kiln_nvtx::range!(c"kiln/norm/pre_attn");
                    rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?
                };
                let attn_out = if stream_active {
                    gated_deltanet_forward_streaming(
                        backend,
                        &normed,
                        lin_weights,
                        config,
                        &mut state.recurrent_states[linear_attn_idx],
                        &mut state.conv_states[linear_attn_idx],
                        stream_tile,
                        layer_lora,
                    )
                    .with_context(|| format!("segment streaming gated deltanet layer {i}"))?
                } else {
                    gated_deltanet_forward(
                        backend,
                        &normed,
                        lin_weights,
                        config,
                        &mut state.recurrent_states[linear_attn_idx],
                        &mut state.conv_states[linear_attn_idx],
                        /* capture_b11_taps = */ false,
                        /* capture_c41_taps = */ false,
                        layer_lora,
                    )
                    .with_context(|| format!("segment gated deltanet layer {i}"))?
                };
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    (hidden + attn_out)?
                };
                let normed_post = {
                    kiln_nvtx::range!(c"kiln/norm/pre_mlp");
                    rms_norm(
                        &hidden,
                        &layer.post_attention_layernorm,
                        config.rms_norm_eps,
                    )?
                };
                let ffn_out = swiglu_ffn(&normed_post, &layer.mlp, layer_lora)?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    (hidden + ffn_out)?
                };
                linear_attn_idx += 1;
            }
        }
    }

    Ok(hidden)
}

/// Compute embedding lookup and add batch dimension.
///
/// Returns `([1, seq_len, hidden_size], positions)` — the initial hidden state
/// and position indices for RoPE (starting from position 0, no KV cache offset).
pub fn model_forward_embed(token_ids: &[u32], weights: &GpuWeights) -> Result<(Tensor, Vec<u32>)> {
    let seq_len = token_ids.len();
    let mut hidden = embedding_lookup_from_weights(token_ids, weights)?;
    hidden = hidden.unsqueeze(0)?;
    let positions: Vec<u32> = (0..seq_len).map(|p| p as u32).collect();
    Ok((hidden, positions))
}

/// Apply final RMSNorm and LM head projection.
///
/// `hidden`: [1, seq_len, hidden_size]
/// Returns: [1, seq_len, vocab_size] logits.
pub fn model_forward_head(
    hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    model_forward_head_backend_decode_if(None, hidden, weights, config)
}

pub fn model_forward_head_backend_decode_if(
    backend: Option<&dyn BackendRuntime>,
    hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    kiln_nvtx::range!(c"kiln/lm_head");
    let normed = rms_norm(hidden, &weights.final_norm, config.rms_norm_eps)?;
    let logits = lm_head_forward_backend_decode_if(backend, &normed, &weights.embed_tokens_t)?;
    Ok(logits)
}

/// Apply only the final RMSNorm (no LM head projection).
///
/// Used by the FLCE training path to produce the post-final-RMSNorm hidden
/// state that `fused_linear_cross_entropy` consumes. Mirrors the RMSNorm
/// step inside [`model_forward_head`] without the vocab-dim matmul.
pub fn model_forward_final_norm(
    hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    kiln_nvtx::range!(c"kiln/final_rmsnorm");
    rms_norm(hidden, &weights.final_norm, config.rms_norm_eps)
}

/// Full training-path forward WITHOUT the LM head projection.
///
/// Runs embedding -> transformer layers -> final RMSNorm, returning the
/// post-final-RMSNorm hidden state `[1, seq_len, hidden_size]`. This is the
/// input the Fused Linear Cross-Entropy path consumes, avoiding the
/// `[1, seq_len, vocab_size]` logits materialization that dominates peak
/// VRAM at long context on the Qwen3.5-4B head (V=151936).
///
/// Call site is the trainer (SFT and GRPO) behind the `KILN_USE_FLCE`
/// environment flag. No KV cache is used (matches `standard_forward_backward`).
pub fn model_forward_no_head(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    let (hidden, positions) = model_forward_embed(token_ids, weights)?;
    let num_layers = weights.layers.len();
    let hidden = model_forward_segment(
        backend,
        hidden,
        weights,
        config,
        &positions,
        0,
        num_layers,
        linear_state,
        lora,
    )?;
    let normed = {
        kiln_nvtx::range!(c"kiln/final_rmsnorm");
        rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?
    };
    Ok(normed)
}

/// Full model forward pass using paged KV cache.
///
/// Same as [`model_forward`] but uses a [`PagedKvCache`] and [`BlockTable`]
/// for KV storage. The caller provides `start_pos` (the absolute position of
/// the first token in `token_ids`) instead of relying on `kv_cache.seq_len()`.
///
/// `positions_gpu`: optional pre-allocated f32 tensor on device with shape [seq_len].
/// When provided, this tensor is used for RoPE instead of creating a new one.
/// This is required for CUDA graph replay: the tensor's GPU address must remain
/// stable so the captured graph reads updated position values on replay.
///
/// Returns logits tensor with shape [1, seq_len, vocab_size].
pub fn model_forward_paged(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<Tensor> {
    let (logits, _hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        positions_gpu,
        #[cfg(feature = "cuda")]
        None,
        LmHeadMode::Full,
    )?;
    // `LmHeadMode::Full` always returns Some.
    Ok(logits.expect("LmHeadMode::Full always produces logits"))
}

/// Paged-KV forward pass for generation prefill when only the next-token
/// distribution is needed.
///
/// This runs the same layer loop and paged KV writes as [`model_forward_paged`]
/// but only projects the final hidden row through the LM head, returning
/// logits with shape `[1, 1, vocab_size]`.
pub fn model_forward_paged_last_token(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<Tensor> {
    let (logits, _hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        positions_gpu,
        #[cfg(feature = "cuda")]
        None,
        LmHeadMode::LastRowOnly,
    )?;
    Ok(logits.expect("LmHeadMode::LastRowOnly always produces logits"))
}

/// Paged-KV forward pass for greedy generation prefill.
///
/// This runs the same prefill work as [`model_forward_paged_last_token`] but
/// fuses the final-row LM-head projection with argmax when the backend supports
/// it, avoiding a `[1, 1, vocab_size]` logits tensor that greedy sampling would
/// immediately reduce.
pub fn model_forward_paged_last_token_greedy(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<u32> {
    let (_logits, _hidden, token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        positions_gpu,
        #[cfg(feature = "cuda")]
        None,
        LmHeadMode::LastRowArgmaxOnly,
    )?;
    token.context("LmHeadMode::LastRowArgmaxOnly always produces a token")
}

/// Paged-KV single-token decode for greedy sampling.
///
/// This keeps the existing logits APIs intact but, on the Metal BF16 decode
/// path, fuses the LM-head projection with argmax so generation does not
/// materialize `[1, 1, vocab_size]` logits only to immediately reduce them.
pub fn model_forward_paged_next_token_greedy(
    backend: &dyn BackendRuntime,
    token_id: u32,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<u32> {
    model_forward_paged_last_token_greedy(
        backend,
        &[token_id],
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        positions_gpu,
    )
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn model_forward_paged_with_graph_inputs(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    token_ids_gpu: &Tensor,
    positions_gpu: &Tensor,
    #[cfg(feature = "cuda")] graph_inputs: Option<&PagedDecodeGraphInputs<'_>>,
) -> Result<Tensor> {
    let (logits, _hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        Some(token_ids_gpu),
        Some(positions_gpu),
        #[cfg(feature = "cuda")]
        graph_inputs,
        LmHeadMode::Full,
    )?;
    Ok(logits.expect("LmHeadMode::Full always produces logits"))
}

/// Batched paged decode forward + LM head argmax with the stable
/// graph inputs threaded through. Lives next to
/// [`model_forward_paged_with_graph_inputs`] but specialized for the
/// `bs > 1` contiguous-batched hot path (the one
/// `ModelRunner::decode_next_tokens_paged_contiguous_batch_greedy_with_ids`
/// drives).
///
/// Today this is a thin stub: it ignores `graph_inputs` and delegates
/// to the eager `model_forward_paged_decode_contiguous_batch_greedy`,
/// which is the same function the existing hot path calls. The
/// captured-batched graph this would feed is not wired in yet. Step 6
/// of the multi-batch capture sequence (see top of `cuda_graph.rs`)
/// replaces the body with a stable-pointer-aware variant that reads
/// from `graph_inputs.token_ids` / `.positions` / etc. and threads
/// the persistent `graph_inputs.linear_state` slot through every
/// GDN layer.
///
/// Returns the per-row next-token IDs (`[batch] u32`), matching the
/// hot path's return shape so the runner's
/// `decode_step_paged_batched` can return `Ok(Some(tokens))`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn model_forward_paged_batched_with_graph_inputs(
    backend: &dyn BackendRuntime,
    input_tokens: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[&BlockTable],
    sequence_lengths: &[usize],
    lora: Option<&LoraWeights>,
    graph_inputs: &mut BatchedPagedDecodeGraphInputs<'_>,
) -> Result<Vec<u32>> {
    // Run the bs>1 hidden path with the persistent linear-state slot
    // and the graph-stable token-id / position device buffers. This is
    // the same code path `decode_next_tokens_paged_contiguous_batch_greedy_with_ids`
    // drives, just with the per-step host→device builds skipped — the
    // captured graph reads from `graph_inputs.token_ids` / `.positions`
    // device pointers that the runner re-fills before each replay.
    let hidden = model_forward_paged_decode_contiguous_batch_hidden_inner(
        backend,
        input_tokens,
        weights,
        config,
        paged_cache,
        block_tables,
        sequence_lengths,
        Some(graph_inputs.linear_state),
        lora,
        Some(graph_inputs.positions),
        Some(graph_inputs.token_ids),
    )?;
    // Compute logits and slice them into the caller-owned stable
    // `output_logits` buffer (`[batch, 1, vocab]`). The captured graph
    // records the matmul + slice_set, so on replay the runner can
    // argmax-reduce the *same* device pointer without re-running any
    // model kernels.
    let normed = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
    let logits = lm_head_forward_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)
        .context("graph-wrapper LM head forward")?;
    graph_inputs
        .output_logits
        .slice_set(&logits, 0, 0)
        .context("copy graph-wrapper logits into stable output_logits buffer")?;
    // Argmax over vocab + DtoH to produce the per-row tokens this
    // call returns. During capture the runner will discard these and
    // re-argmax `output_logits` after every replay — but returning
    // them here keeps the function's behavior consistent across
    // capture and direct-call use.
    let tokens = {
        kiln_nvtx::range!(c"kiln/lm_head_batch_argmax_decode_graph");
        lm_head_argmax_rows_backend_decode_if(
            Some(backend),
            &normed,
            &weights.embed_tokens_t,
        )?
    };
    Ok(tokens)
}

/// Batched paged decode API for real continuous-batching work.
///
/// Keeps the existing [`PagedKvCache`] API and its caller-held mutex: each
/// request still has its own [`BlockTable`] and KV window, but the dominant
/// GDN/MLP layers run as one batch-shaped forward. Full-attention layers stay
/// row-wise because each request has distinct paged KV metadata; this avoids
/// the batch-8 paged-attention workspace blow-up while still removing the 24
/// GDN-layer row loop that made streaming throughput flat.
///
/// CUDA graphs are deliberately not used for `batch_size > 1` here; the graph
/// runner is currently captured for the batch-1 decode shape only. TODO(phase2
/// continuous batching): add graph capture/replay keyed by decode batch shape.
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_batched_decode(
    backend: &dyn BackendRuntime,
    input_tokens: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[BlockTable],
    sequence_lengths: &[usize],
    linear_states: &mut [&mut LinearAttentionState],
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    let hidden = model_forward_paged_batched_decode_hidden(
        backend,
        input_tokens,
        weights,
        config,
        paged_cache,
        block_tables,
        sequence_lengths,
        linear_states,
        lora,
    )?;
    model_forward_head_backend_decode_if(Some(backend), &hidden, weights, config)
        .context("batched decode lm head")
}

/// Batched paged decode through the transformer stack, stopping before the
/// final LM head.
///
/// Returning `[batch, 1, hidden]` lets the caller choose between projecting the
/// whole batch with a backend-aware LM head or sampling rows independently when
/// bounded LM-head workspace is more important than projection throughput.
#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_batched_decode_hidden(
    backend: &dyn BackendRuntime,
    input_tokens: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_tables: &[BlockTable],
    sequence_lengths: &[usize],
    linear_states: &mut [&mut LinearAttentionState],
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    let batch_size = input_tokens.len();
    anyhow::ensure!(batch_size > 0, "batched decode requires at least one token");
    anyhow::ensure!(
        block_tables.len() == batch_size,
        "batched decode block_tables length {} != input_tokens length {batch_size}",
        block_tables.len()
    );
    anyhow::ensure!(
        sequence_lengths.len() == batch_size,
        "batched decode sequence_lengths length {} != input_tokens length {batch_size}",
        sequence_lengths.len()
    );
    anyhow::ensure!(
        linear_states.len() == batch_size,
        "batched decode linear_states length {} != input_tokens length {batch_size}",
        linear_states.len()
    );

    if batch_size == 1 {
        let (_, hidden, _) = model_forward_paged_inner(
            backend,
            &[input_tokens[0]],
            weights,
            config,
            paged_cache,
            &block_tables[0],
            sequence_lengths[0],
            Some(&mut *linear_states[0]),
            lora,
            None,
            None,
            #[cfg(feature = "cuda")]
            None,
            LmHeadMode::HiddenOnly,
        )?;
        return hidden.context("batched decode hidden skipped lm head");
    }

    let device = weights.embed_tokens.device();
    let mut hidden = embedding_lookup_from_weights(input_tokens, weights)?;
    hidden = hidden.unsqueeze(1)?;
    let use_metal_decode_ffn = sequence_lengths.iter().all(|&p| p > 0)
        && !crate::mtp_debug::is_mtp_single_token_self_attn_armed();

    let mut full_attn_idx = 0usize;
    let mut linear_attn_idx = 0usize;
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let layer_lora: Option<(&LoraLayerWeights, f32)> =
            lora.and_then(|lw| lw.layers.get(layer_idx).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Linear(lin_weights) => {
                let normed = {
                    kiln_nvtx::range!(c"kiln/batched_decode/norm/pre_attn");
                    rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?
                };

                // Defensive dtype normalization (same rationale as
                // LinearAttentionState::from_batch_rows): cast any drifted rows
                // back to row 0's dtype before cat, so a stray BF16 row from a
                // prior aborted decode does not break the slow path either.
                let mut recurrent_state = {
                    let target_dtype = linear_states[0].recurrent_states[linear_attn_idx].dtype();
                    let mut owned: Vec<Tensor> = Vec::with_capacity(linear_states.len());
                    for (row_idx, state) in linear_states.iter().enumerate() {
                        let t = &state.recurrent_states[linear_attn_idx];
                        if t.dtype() != target_dtype {
                            tracing::debug!(
                                layer = layer_idx,
                                row = row_idx,
                                from = ?t.dtype(),
                                to = ?target_dtype,
                                "batched_decode: normalizing recurrent state dtype before cat"
                            );
                            owned.push(t.to_dtype(target_dtype).with_context(|| {
                                format!(
                                    "cast recurrent state row {row_idx} to {target_dtype:?} for GDN layer {layer_idx}"
                                )
                            })?);
                        } else {
                            owned.push(t.clone());
                        }
                    }
                    let refs: Vec<&Tensor> = owned.iter().collect();
                    Tensor::cat(&refs, 0).with_context(|| {
                        format!("cat batched recurrent state for GDN layer {layer_idx}")
                    })?
                };
                let mut conv_state = {
                    let target_dtype = linear_states[0].conv_states[linear_attn_idx].dtype();
                    let mut owned: Vec<Tensor> = Vec::with_capacity(linear_states.len());
                    for (row_idx, state) in linear_states.iter().enumerate() {
                        let t = &state.conv_states[linear_attn_idx];
                        if t.dtype() != target_dtype {
                            tracing::debug!(
                                layer = layer_idx,
                                row = row_idx,
                                from = ?t.dtype(),
                                to = ?target_dtype,
                                "batched_decode: normalizing conv state dtype before cat"
                            );
                            owned.push(t.to_dtype(target_dtype).with_context(|| {
                                format!(
                                    "cast conv state row {row_idx} to {target_dtype:?} for GDN layer {layer_idx}"
                                )
                            })?);
                        } else {
                            owned.push(t.clone());
                        }
                    }
                    let refs: Vec<&Tensor> = owned.iter().collect();
                    Tensor::cat(&refs, 0).with_context(|| {
                        format!("cat batched conv state for GDN layer {layer_idx}")
                    })?
                };

                let attn_out = gated_deltanet_forward_decode_if(
                    backend,
                    &normed,
                    lin_weights,
                    config,
                    &mut recurrent_state,
                    &mut conv_state,
                    false,
                    false,
                    true,
                    false,
                    None,
                    true,
                    true,
                    layer_lora,
                )
                .with_context(|| format!("batched GDN layer {layer_idx}"))?;

                for (row_idx, state) in linear_states.iter_mut().enumerate() {
                    state.recurrent_states[linear_attn_idx] = recurrent_state
                        .narrow(0, row_idx, 1)?
                        .contiguous()
                        .with_context(|| {
                            format!("split recurrent state row {row_idx} for GDN layer {layer_idx}")
                        })?;
                    state.conv_states[linear_attn_idx] = conv_state
                        .narrow(0, row_idx, 1)?
                        .contiguous()
                        .with_context(|| {
                            format!("split conv state row {row_idx} for GDN layer {layer_idx}")
                        })?;
                }

                hidden = {
                    kiln_nvtx::range!(c"kiln/batched_decode/residual/attn");
                    (hidden + attn_out)?
                };
                let normed_post = {
                    kiln_nvtx::range!(c"kiln/batched_decode/norm/pre_mlp");
                    rms_norm(
                        &hidden,
                        &layer.post_attention_layernorm,
                        config.rms_norm_eps,
                    )?
                };
                let ffn_out = swiglu_ffn_backend_profiled(
                    backend,
                    &normed_post,
                    &layer.mlp,
                    layer_lora,
                    use_metal_decode_ffn,
                    None,
                )?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/batched_decode/residual/mlp");
                    (hidden + ffn_out)?
                };
                linear_attn_idx += 1;
            }
            GpuAttentionWeights::Full(_) => {
                let positions_f32: Vec<f32> = sequence_lengths.iter().map(|&p| p as f32).collect();
                let positions = Tensor::from_slice(positions_f32.as_slice(), batch_size, device)?;
                let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
                match transformer_block_paged_decode_contiguous_batch(
                    backend,
                    &hidden,
                    layer,
                    config,
                    &positions,
                    sequence_lengths,
                    &weights.rotary_inv_freq,
                    paged_cache,
                    &block_table_refs,
                    full_attn_idx,
                    layer_lora,
                    None,
                    None,
                    None,
                ) {
                    Ok(out) => hidden = out,
                    Err(err) => {
                        tracing::debug!(
                            layer = layer_idx,
                            error = %err,
                            "batched full-attention decode declined; falling back to rowwise"
                        );
                        let mut rows = Vec::with_capacity(batch_size);
                        for row_idx in 0..batch_size {
                            let row_hidden = hidden.narrow(0, row_idx, 1)?.contiguous()?;
                            let row_position = Tensor::from_slice(
                                &[sequence_lengths[row_idx] as f32],
                                1usize,
                                device,
                            )?;
                            let row = transformer_block_paged(
                                backend,
                                &row_hidden,
                                layer,
                                config,
                                &row_position,
                                sequence_lengths[row_idx],
                                config.num_attention_heads,
                                config.num_kv_heads,
                                config.head_dim,
                                config.rotary_dim(),
                                &weights.rotary_inv_freq,
                                config.rms_norm_eps,
                                paged_cache,
                                &block_tables[row_idx],
                                full_attn_idx,
                                layer_lora,
                            )
                            .with_context(|| {
                                format!(
                                    "rowwise fallback transformer block {layer_idx} row {row_idx} (full attention, paged)"
                                )
                            })?;
                            rows.push(row);
                        }
                        let row_refs: Vec<&Tensor> = rows.iter().collect();
                        hidden = Tensor::cat(&row_refs, 0).with_context(|| {
                            format!("cat rowwise fallback transformer block {layer_idx} outputs")
                        })?;
                    }
                }
                full_attn_idx += 1;
            }
        }
    }

    Ok(hidden)
}

/// Paged-KV forward pass that ALSO returns the last-row pre-final-norm hidden state.
///
/// Same semantics as [`model_forward_paged`] (identical layer loop, RoPE,
/// paged KV writes), but extracts the last token's hidden state BEFORE
/// `final_norm` is applied. This is the `h_prev` input the native MTP head
/// consumes for speculative decoding: see [`mtp_forward_step`].
///
/// Returns `(logits[1, seq_len, V], hidden_last[1, 1, H])`. Logits are
/// returned per-position so MTP speculative verification can compare the
/// draft token against position 0 (`logits[:, 0, :]` predicts what should
/// follow the last committed token) and sample a bonus token from position
/// `seq_len - 1` on full acceptance.
pub fn model_forward_paged_with_last_hidden(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    // Phase B10: arm the base-model per-layer hidden-state capture window
    // when `KILN_MTP_DUMP_HIDDEN_STATES=1`. The arm is a no-op when the env
    // var is unset, so production cost is a single TLS borrow + env lookup.
    // The inner forward pass fills the window with boundary-layer last-row
    // slices plus `h_post_final_norm` (C18 — formerly `h_pre_final_norm`
    // before kiln started returning post-final-norm `h_prev`). Phase C40
    // can opt into a denser early sweep (layers 1..8) via
    // `KILN_MTP_DUMP_EARLY_HMAIN_SWEEP=1`; default B10/B12 behavior is
    // unchanged when that flag is unset. The window is drained in
    // `mtp_forward_step`'s dump block so the taps appear alongside the
    // standard 8 MTP taps in the same safetensors file. The next call to
    // this function re-arms the window, overwriting any stale buffer from
    // a prior call whose dump did not fire (e.g. non-targeted `mtp_pos`).
    crate::mtp_debug::arm_h_main_capture();
    // Phase B11: stash the exact input tokens that fed this forward pass so
    // the MTP dump can serialize them, letting the HF reference replay the
    // same prompt instead of its canonical fallback greeting. No-op when
    // h_main capture is disarmed.
    crate::mtp_debug::stash_h_main_replay_context(token_ids);
    // Phase B11b: arm the layer-0 GDN sub-op capture window in the same
    // place as h_main so both capture modes drain together inside the MTP
    // dump block. No-op unless `KILN_MTP_DUMP_B11_TAPS=1`, so production
    // decode pays only a single TLS borrow + env-var lookup.
    crate::mtp_debug::arm_b11_layer0_capture();
    // Phase B12: arm the layer-31 GQA sub-op capture window. Same pattern
    // as B11 — no-op unless `KILN_MTP_DUMP_B12_GQA_TAPS=1`. The h_main
    // capture is gated to also include layers 24..30 when this flag is on,
    // giving the comparator both per-layer h_layer_<idx> taps for the GQA
    // tail and per-sub-op taps inside layer 31.
    crate::mtp_debug::arm_b12_gqa_capture();
    crate::mtp_debug::arm_c41_layer1_capture();
    crate::mtp_debug::arm_c42_layer1_norm_capture();
    crate::mtp_debug::arm_c43_layer1_preweight_capture();
    crate::mtp_debug::arm_c44_layer1_f32_row_capture();
    crate::mtp_debug::arm_c45_layer1_row_capture();
    crate::mtp_debug::arm_c46_layer1_row_provenance_capture();
    let (logits, hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        positions_gpu,
        #[cfg(feature = "cuda")]
        None,
        LmHeadMode::FullWithLastHidden,
    )?;
    Ok((
        logits.expect("LmHeadMode::FullWithLastHidden always produces logits"),
        hidden.expect("LmHeadMode::FullWithLastHidden always produces hidden"),
    ))
}

/// Paged-KV forward pass for MTP prefill.
///
/// Returns only the last-row logits plus the last-row pre-final-norm hidden
/// state. MTP prefill does not need per-position logits, so this avoids
/// projecting every prompt row through the large tied LM head.
pub fn model_forward_paged_last_token_with_last_hidden(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    crate::mtp_debug::arm_h_main_capture();
    crate::mtp_debug::stash_h_main_replay_context(token_ids);
    crate::mtp_debug::arm_b11_layer0_capture();
    crate::mtp_debug::arm_c41_layer1_capture();
    crate::mtp_debug::arm_c42_layer1_norm_capture();
    crate::mtp_debug::arm_c43_layer1_preweight_capture();
    crate::mtp_debug::arm_c44_layer1_f32_row_capture();
    crate::mtp_debug::arm_c45_layer1_row_capture();
    crate::mtp_debug::arm_c46_layer1_row_provenance_capture();
    let (logits, hidden, _token) = model_forward_paged_inner(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
        positions_gpu,
        #[cfg(feature = "cuda")]
        None,
        LmHeadMode::LastRowWithLastHidden,
    )?;
    Ok((
        logits.expect("LmHeadMode::LastRowWithLastHidden always produces logits"),
        hidden.expect("LmHeadMode::LastRowWithLastHidden always produces hidden"),
    ))
}

/// Single-step native MTP (Multi-Token Prediction) forward pass.
///
/// Implements the Qwen3-Next-style MTP head described in the vLLM reference
/// (`qwen3_next_mtp.py`): given the previously generated token and the base
/// model's pre-final-norm hidden state, project them through the MTP fusion
/// layer and a single full-attention transformer block to produce logits for
/// the NEXT token, plus an updated hidden state that can be fed back for
/// multi-step drafting (when `num_nextn_predict_layers > 1`; Qwen3.5-4B ships
/// `k=1` so drafts are exactly one token deep).
///
/// Fusion pipeline:
///
/// 1. `token_emb  = embed_tokens[draft_token_id]`   # [1, 1, H]
/// 2. `norm_emb   = rms_norm(token_emb, pre_fc_norm_embedding)`
/// 3. `norm_h     = rms_norm(h_prev,    pre_fc_norm_hidden)`
/// 4. `fused      = concat([norm_emb, norm_h], dim=-1) @ fc_t`   # [1,1,2H]→[1,1,H]
/// 5. `hidden     = transformer_block_paged(mtp_layer, fused, mtp_cache, mtp_pos)`
/// 6. `logits     = rms_norm(hidden, final_layernorm) @ embed_tokens_t`  # tied head
///
/// Returns `(logits[1,1,V], new_hidden[1,1,H])`. `new_hidden` is the
/// pre-final-norm output of the MTP transformer block and is the `h_prev`
/// input for the next MTP step (unused when k=1).
///
/// ## KV cache discipline
///
/// The MTP layer maintains its own `PagedKvCache` with exactly ONE full-attn
/// layer slot. `mtp_pos` is the absolute position at which to write this
/// step's KV. Callers advance `mtp_pos` by +1 ONLY when the draft token is
/// accepted; on rejection `mtp_pos` stays unchanged and the next call
/// overwrites the just-written KV slot (the paged writes are idempotent at a
/// given position, so rejection is implicit — no explicit rollback needed).
///
/// ## Marlin / LoRA
///
/// The MTP layer is NOT currently Marlin-packed (deferred to a follow-up PR —
/// Marlin adds substantial pack latency at model load and the MTP layer is a
/// small fraction of per-step cost). LoRA is not applied to MTP.
#[allow(clippy::too_many_arguments)]
pub fn mtp_forward_step(
    backend: &dyn BackendRuntime,
    draft_token_id: u32,
    h_prev: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    mtp_cache: &PagedKvCache,
    mtp_block_table: &BlockTable,
    base_pos: usize,
    mtp_pos: usize,
) -> Result<(Tensor, Tensor)> {
    kiln_nvtx::range!(c"kiln/mtp/step");
    let mtp = weights.mtp_weights()?;
    let device = weights.embed_tokens.device();

    // Phase C6 dump pre-flight: consume the dump slot up-front so we can arm
    // the pre-RoPE capture window BEFORE any of the 5 pre-RoPE tensors
    // (token_emb, norm_emb, norm_h, concat, fused) are materialized. The slot
    // is one-shot per (process, mtp_pos), so `should_dump` threads through to
    // the dump block below without being re-consumed. When
    // `KILN_MTP_DUMP_PRE_ROPE` is unset the arm is a no-op and the per-tap
    // `capture_pre_rope_tap` calls short-circuit on a closed TLS window.
    // Phase C13: `KILN_MTP_DUMP_SPLICE=1` is a meta-flag that fires up to N=8
    // (configurable) dumps per targeted position (default `{0, 2}`) instead of
    // the one-shot latch used by earlier phases. When the splice lane takes a
    // slot, `splice_step` is `Some(step)` and the dump path can substitute
    // `{step}` alongside `{pos}`. When splice is disabled we fall through to
    // the existing one-shot latch so prior flows (B6/B7/C6/C7) behave
    // unchanged.
    let splice_step = crate::mtp_debug::try_consume_splice_slot(mtp_pos);
    let should_dump = if splice_step.is_some() {
        true
    } else if crate::mtp_debug::is_dump_splice_enabled() {
        // Splice is on but this position/step is not eligible — suppress the
        // legacy one-shot latch so only the splice lane controls dumping.
        false
    } else {
        crate::mtp_debug::try_consume_dump_slot_for_pos(mtp_pos)
    };
    let dump_pre_rope = should_dump && crate::mtp_debug::is_dump_pre_rope_effectively_enabled();
    if dump_pre_rope {
        crate::mtp_debug::arm_pre_rope_capture();
    }
    // Phase C7 dump pre-flight: arm the SDPA-internal capture BEFORE the
    // inner transformer block runs, so the 7 taps inside `gqa_attention_paged`
    // can record Q/K/V, scores, probs, and the raw attention output. Armed
    // independently of C6 because the two capture windows bracket different
    // regions of the forward: C6 captures MTP fc inputs pre-RoPE, C7
    // captures SDPA inside the inner block post-RoPE. Arming C7 also acts as
    // a signal to the GQA path to bypass the fused flash-attention paged
    // decode kernel (which doesn't materialize the intermediates we need)
    // and take the unfused grouped-decode Candle path instead.
    let dump_c7_sdpa = should_dump && crate::mtp_debug::is_dump_c7_sdpa_enabled();
    if dump_c7_sdpa {
        crate::mtp_debug::arm_c7_sdpa_capture();
    }

    // Phase C14 post-block splice-dump pre-flight. Mirrors the C7 arm above.
    // Gated on `should_dump` AND either the explicit
    // `KILN_MTP_DUMP_C14_POST_BLOCK=1` opt-in or (via OR-composition inside
    // `is_dump_c14_post_block_effectively_enabled`) the C13 splice meta-flag
    // `KILN_MTP_DUMP_SPLICE=1`. When armed, we capture three taps after the
    // MTP transformer block returns: `post_block` (pre-norm hidden),
    // `post_norm` (post-final-norm hidden, pre-lm_head), and `logits`
    // (post-lm_head, pre-softmax). This is the extension of the splice
    // window past the `c6__fused` exit that C13 certified clean.
    let dump_c14_post_block =
        should_dump && crate::mtp_debug::is_dump_c14_post_block_effectively_enabled();
    if dump_c14_post_block {
        crate::mtp_debug::arm_c14_post_block_capture();
    }

    // 1. Token embedding for the draft token. `embedding_lookup` returns
    //    shape [1, H]; unsqueeze to [1, 1, H] to match transformer-block I/O.
    let token_ids = [draft_token_id];
    let token_emb = embedding_lookup_from_weights(&token_ids, weights)?; // [1, H]
    let token_emb = token_emb.unsqueeze(0)?; // [1, 1, H]
    if dump_pre_rope {
        let _ = crate::mtp_debug::capture_pre_rope_tap("token_emb", &token_emb);
    }

    // 2-3. Dual RMSNorms. `h_prev` is [1, 1, H] pre-final-norm.
    //
    // `KILN_MTP_SWAP_FC_NORMS=1` swaps which RMSNorm weight is applied to
    // which half. This is the Phase B2 secondary-hypothesis A/B: if the
    // loader paired the two `pre_fc_norm_*` tensors to the wrong halves of
    // the `fc` input (plausible since both are [H]-vectors and
    // distinguishable only by name), swap-on should materially change α.
    // If α is unchanged the hypothesis is disproven.
    let swap_fc_norms = crate::mtp_debug::is_swap_fc_norms_enabled();
    let (norm_emb_weight, norm_h_weight) = if swap_fc_norms {
        (&mtp.pre_fc_norm_hidden, &mtp.pre_fc_norm_embedding)
    } else {
        (&mtp.pre_fc_norm_embedding, &mtp.pre_fc_norm_hidden)
    };
    let norm_emb = {
        kiln_nvtx::range!(c"kiln/mtp/pre_fc_norm_emb");
        rms_norm(&token_emb, norm_emb_weight, config.rms_norm_eps)?
    };
    if dump_pre_rope {
        let _ = crate::mtp_debug::capture_pre_rope_tap("norm_emb", &norm_emb);
    }
    let norm_h = {
        kiln_nvtx::range!(c"kiln/mtp/pre_fc_norm_hidden");
        rms_norm(h_prev, norm_h_weight, config.rms_norm_eps)?
    };
    if dump_pre_rope {
        let _ = crate::mtp_debug::capture_pre_rope_tap("norm_h", &norm_h);
    }

    // 4. Concat along the hidden dim and fuse: [1, 1, 2H] @ fc_t[2H, H] -> [1, 1, H]
    //
    // We keep the concat alive (named `concat`) so the Phase B6 dump can
    // capture the exact bytes fed into `fc.weight` as `fc_input`.
    let concat = Tensor::cat(&[&norm_emb, &norm_h], 2)?.contiguous()?;
    if dump_pre_rope {
        let _ = crate::mtp_debug::capture_pre_rope_tap("concat", &concat);
    }
    // Phase C12: `KILN_MTP_FP32_HEAD=1` subsumes `KILN_MTP_FC_FP32_ACCUM=1` —
    // the full-head kill switch always includes fc_input/fc_output in f32,
    // so either flag alone is sufficient to trigger the fp32 fc path.
    let fp32_head = crate::mtp_debug::is_mtp_fp32_head_enabled();
    let fused = {
        kiln_nvtx::range!(c"kiln/mtp/fc");
        if crate::mtp_debug::is_mtp_fc_fp32_accum_enabled() || fp32_head {
            // Phase C9 falsification: promote inputs to f32, matmul in f32,
            // cast the result back to the input dtype. Eliminates the bf16
            // accumulation noise visible at the `fused` / `fc_output` tap
            // (max|Δ| ~1.6e-2 against the HF bf16 reference). The
            // [1, 1, 2H] @ [2H, H] shape is tiny (~13M FLOPs for 4B), so
            // the per-step cost is negligible and there is no hot-path
            // regression to worry about.
            let in_dtype = concat.dtype();
            let concat_f32 = concat.to_dtype(candle_core::DType::F32)?;
            let fc_t_f32 = mtp.fc_t.to_dtype(candle_core::DType::F32)?;
            concat_f32.broadcast_matmul(&fc_t_f32)?.to_dtype(in_dtype)?
        } else {
            concat.broadcast_matmul(&mtp.fc_t)?
        }
    };
    if dump_pre_rope {
        let _ = crate::mtp_debug::capture_pre_rope_tap("fused", &fused);
    }

    // Phase B7 sub-op capture pre-flight. `should_dump` was already consumed
    // above (moved up to bracket Phase C6 pre-RoPE arming). `dump_subops` is
    // a strict subset: only true when KILN_MTP_DUMP_SUBOPS=1 AND we are about
    // to dump anyway. This keeps the production path entirely free of sub-op
    // capture overhead — the TLS check inside `capture_subop` is a no-op when
    // the window is closed.
    let dump_subops = should_dump && crate::mtp_debug::is_dump_subops_enabled();
    if dump_subops {
        crate::mtp_debug::arm_subop_capture();
    }

    // 5. Single full-attention transformer block with its own paged cache.
    //
    //    Two distinct position counters are in play here:
    //
    //    * `base_pos + mtp_pos` — the ABSOLUTE sequence position the draft
    //      token would occupy in the prompt+decode stream. This is what
    //      RoPE must use so the MTP head sees the same rotation angles the
    //      base Qwen3-Next block would have applied at that position. The
    //      PyTorch reference (`scripts/mtp_reference_dump.py`) applies RoPE
    //      at the absolute position; Phase B7a (PR #276) confirmed kiln's
    //      prior use of bare `mtp_pos` here caused monotonic `post_layer`
    //      drift at pos=1,2 — the RoPE-wrong-position signature.
    //
    //    * `mtp_pos` — the LOCAL slot index into the MTP paged KV cache.
    //      The MTP cache is its own isolated address space (distinct from
    //      the base KV cache); slot `mtp_pos` is the right write target
    //      regardless of where the token sits in absolute stream order.
    //
    //    MTP is not CUDA-graph-captured, so rebuilding the position tensor
    //    per step is fine.
    let abs_pos = base_pos + mtp_pos;
    let positions = Tensor::new(&[abs_pos as f32][..], device)?;
    // Phase C8: arm single-token self-attention for the MTP inner GQA call.
    // The Qwen3-Next reference contract (see `scripts/mtp_reference_dump.py`
    // and HF/vLLM `Qwen3NextMultiTokenPredictor`) runs the MTP inner
    // attention as kv_len = 1 — Q·K^T is a 1×1 scalar, softmax = 1.0,
    // attn_out = V. Phase C7 (PR #319) localized the mtp_pos > 0
    // attn_out divergence to kiln attending over the growing MTP paged
    // cache (kv_len = mtp_pos + 1) instead of single-token self-attn.
    // Arming this flag flips `gqa_attention_paged` onto the per-step
    // K/V scratch path for the MTP layer only; the disarm below clears
    // it before any non-MTP attention path on this thread can observe it.
    crate::mtp_debug::arm_mtp_single_token_self_attn();
    // Phase C12: arm fp32-head BEFORE the inner block so that every
    // projection matmul inside `gqa_attention_paged` (q/k/v/o) and the
    // MLP (`swiglu_ffn`'s gate/up/down) sees the TLS flag armed. This is
    // the cleanest minimal invasive cast point: `linear_with_lora_t` is
    // the single chokepoint for all of those matmuls, and the non-MTP
    // paths never observe the armed flag because it is disarmed below
    // before we return. The MTP head is not Marlin-packed today, so in
    // practice the flag gates the straight BF16 `broadcast_matmul`; if a
    // future PR adds Marlin to MTP, the marlin path in `q_proj_forward`
    // will need an analogous upcast branch.
    if fp32_head {
        crate::mtp_debug::arm_mtp_fp32_head();
    }
    let mtp_hidden_result = transformer_block_paged(
        backend,
        &fused,
        &mtp.layer,
        config,
        &positions,
        mtp_pos,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.rotary_dim(),
        &weights.rotary_inv_freq,
        config.rms_norm_eps,
        mtp_cache,
        mtp_block_table,
        /* full_attn_layer_idx = */ 0,
        /* lora = */ None,
    );
    // Always disarm in both success and error paths (`mtp_hidden_result`
    // is `?`-propagated below, so we cannot rely on the function tail).
    crate::mtp_debug::disarm_mtp_single_token_self_attn();
    if fp32_head {
        crate::mtp_debug::disarm_mtp_fp32_head();
    }

    // Drain the sub-op capture window in BOTH success and error paths so the
    // TLS slot is not left armed for the next draft step (which would corrupt
    // the next dump or leak captures into other transformer block calls).
    //
    // Phase B10 appends per-layer base-model hidden-state taps captured during
    // the prior `model_forward_paged_with_last_hidden` call on this thread.
    // These live in a distinct TLS slot (`H_MAIN_CAPTURE`) gated on
    // `KILN_MTP_DUMP_HIDDEN_STATES=1`; `drain_h_main_capture` returns empty
    // when the slot was never armed, so disarmed runs pay zero cost.
    let mut extra_subops = if dump_subops {
        crate::mtp_debug::drain_subop_capture()
    } else {
        Vec::new()
    };
    extra_subops.extend(crate::mtp_debug::drain_h_main_capture());
    let mtp_hidden = mtp_hidden_result.context("mtp transformer block")?;
    // Phase C14 tap 1/3: pre-final-norm output of the MTP transformer block.
    if dump_c14_post_block {
        let _ = crate::mtp_debug::capture_c14_post_block_tap("post_block", &mtp_hidden);
    }

    // 6. Final RMSNorm + weight-tied LM head (reuses base embed_tokens_t).
    //
    // We split `normed` out as a distinct bind (rather than inlining into the
    // `logits` block) so the Phase B6 dump can capture `post_final_ln` ahead
    // of the `lm_head` matmul. No semantic change vs the previous inlined
    // form: `rms_norm` has no side effects, and `normed` is only used once.
    let normed = {
        kiln_nvtx::range!(c"kiln/mtp/final_layernorm");
        rms_norm(&mtp_hidden, &mtp.final_layernorm, config.rms_norm_eps)?
    };
    // Phase C14 tap 2/3: post-final-norm hidden state, pre-lm_head.
    if dump_c14_post_block {
        let _ = crate::mtp_debug::capture_c14_post_block_tap("post_norm", &normed);
    }
    let logits = {
        kiln_nvtx::range!(c"kiln/mtp/lm_head");
        lm_head_forward(&normed, &weights.embed_tokens_t)?
    };
    // Phase C14 tap 3/3: post-lm_head logits, pre-softmax / pre-sampler.
    if dump_c14_post_block {
        let _ = crate::mtp_debug::capture_c14_post_block_tap("logits", &logits);
    }

    // Phase B6/B7 numerical-bisect dump. Fires once per (process, mtp_pos)
    // pair when `KILN_MTP_DUMP_PATH` is set and the current `mtp_pos` is
    // listed in `KILN_MTP_DUMP_POS` (defaults to "0" for B6 compatibility).
    // Writes one safetensors file per targeted position with the 8 outer
    // taps enumerated in `write_mtp_dump` plus integer metadata (draft
    // token id, `mtp_pos`, `swap_fc_norms`). When `KILN_MTP_DUMP_SUBOPS=1`
    // is also set, per-sub-op activations from inside the MTP transformer
    // block are appended (Phase B7b).
    //
    // Use `KILN_MTP_DUMP_PATH=/path/dump_pos{pos}.st` plus
    // `KILN_MTP_DUMP_POS=0,1,2` to capture three positions in one process.
    // The companion Python reference (`scripts/mtp_reference_dump.py`)
    // produces same-shaped files for the same prompt + seed;
    // `scripts/mtp_compare.py` prints a per-tap first-divergence table.
    // Failure to dump is logged but non-fatal — we never want an
    // instrumentation bug to break decode.
    if should_dump {
        // Phase C13: when the splice meta-flag is driving this step, the path
        // can substitute `{step}` alongside `{pos}` so each of the up-to-8
        // per-position dumps lands in its own file. Falls back to the legacy
        // `{pos}`-only substitution when splice is off.
        let dump_path_opt = crate::mtp_debug::dump_path_for_pos_and_step(mtp_pos, splice_step);
        // Always drain the C7 TLS slot before returning. If we entered the
        // armed C7 path above but `dump_path_for_pos` returned None (pos not
        // listed in `KILN_MTP_DUMP_POS`), the slot would otherwise leak
        // captured tensors into the next draft step's dump. Dropping the
        // drained vec here is cheap and keeps the invariant "armed ⇒ drained
        // within the same mtp_forward_step".
        if dump_c7_sdpa && dump_path_opt.is_none() {
            let _ = crate::mtp_debug::drain_c7_sdpa_capture();
        }
        // Phase C14: mirror the C7 defensive drain so the post-block TLS slot
        // is not left armed for the next draft step when the path is filtered
        // out for this pos.
        if dump_c14_post_block && dump_path_opt.is_none() {
            let _ = crate::mtp_debug::drain_c14_post_block_capture();
        }
        if let Some(path) = dump_path_opt {
            let taps: [(&str, &Tensor); 8] = [
                ("h_main", h_prev),
                ("tok_embed", &token_emb),
                ("fc_input", &concat),
                ("fc_output", &fused),
                ("pre_layer", &fused),
                ("post_layer", &mtp_hidden),
                ("post_final_ln", &normed),
                ("mtp_logits", &logits),
            ];
            // Phase B11: drain any prompt tokens stashed by the preceding
            // `model_forward_paged_with_last_hidden` call. Empty on legacy
            // paths / when h_main capture was never armed, which matches
            // the pre-B11 dump format (no `prompt_tokens` tensor emitted).
            let prompt_tokens = crate::mtp_debug::drain_h_main_prompt_tokens();
            let replay_tokens = crate::mtp_debug::drain_h_main_replay_tokens();
            // Phase B11b: drain any layer-0 GDN sub-op taps stashed during
            // the base-model forward. Empty when `KILN_MTP_DUMP_B11_TAPS`
            // is unset, which keeps the dump format bit-identical to B11.
            let b11_taps = crate::mtp_debug::drain_b11_layer0_capture();
            // Phase B12: drain any layer-31 GQA sub-op taps stashed during
            // the base-model forward. Empty when
            // `KILN_MTP_DUMP_B12_GQA_TAPS` is unset, which keeps the dump
            // format bit-identical to the pre-B12 layout.
            let b12_taps = crate::mtp_debug::drain_b12_gqa_capture();
            // Phase C41: drain any transformer-block-1 taps stashed during
            // the base-model forward. Empty when
            // `KILN_MTP_DUMP_C41_LAYER1_TAPS` is unset.
            let c41_taps = crate::mtp_debug::drain_c41_layer1_capture();
            // Phase C42: drain any layer-1 pre-norm / input-layernorm taps
            // stashed during the base-model forward. Empty when
            // `KILN_MTP_DUMP_C42_LAYER1_NORM_TAPS` is unset.
            let c42_taps = crate::mtp_debug::drain_c42_layer1_norm_capture();
            // Phase C43: drain the layer-1 pre-weight multiply taps stashed
            // during the base-model forward. Empty when
            // `KILN_MTP_DUMP_C43_LAYER1_PREWEIGHT_TAPS` is unset.
            let c43_taps = crate::mtp_debug::drain_c43_layer1_preweight_capture();
            // Phase C44: drain the row-level layer-1 taps stashed during the
            // base-model forward. Empty when
            // `KILN_MTP_DUMP_C44_LAYER1_F32_ROW_TAPS` is unset.
            let c44_taps = crate::mtp_debug::drain_c44_layer1_f32_row_capture();
            // Phase C45: drain the follow-up row-level normalization taps
            // stashed during the base-model forward. Empty when
            // `KILN_MTP_DUMP_C45_LAYER1_ROW_TAPS` is unset.
            let c45_taps = crate::mtp_debug::drain_c45_layer1_row_capture();
            // Phase C46: drain the C45 row-side operand provenance taps
            // stashed during the base-model forward. Empty when
            // `KILN_MTP_DUMP_C46_ROW_PROVENANCE` is unset.
            let c46_taps = crate::mtp_debug::drain_c46_layer1_row_provenance_capture();
            // Phase C6: drain the 5 pre-RoPE MTP input taps (token_emb,
            // norm_emb, norm_h, concat, fused) captured above. Empty when
            // `KILN_MTP_DUMP_PRE_ROPE` is unset, which keeps the dump format
            // bit-identical to the pre-C6 layout.
            let c6_taps = crate::mtp_debug::drain_pre_rope_capture();
            // Phase C7: drain the 7 SDPA-internal taps (pre_sdpa_q/k/v,
            // causal_mask, attn_scores_pre_softmax, attn_probs, attn_out)
            // captured inside `gqa_attention_paged`. Empty when
            // `KILN_MTP_DUMP_C7_SDPA` is unset, which keeps the dump format
            // bit-identical to the pre-C7 layout.
            let c7_taps = crate::mtp_debug::drain_c7_sdpa_capture();
            // Phase C14: drain the 3 post-MTP-transformer-block taps
            // (post_block, post_norm, logits) captured above. Empty when
            // neither `KILN_MTP_DUMP_C14_POST_BLOCK` nor the C13 splice
            // meta-flag is set, which keeps the dump format bit-identical
            // to the pre-C14 layout.
            let c14_taps = crate::mtp_debug::drain_c14_post_block_capture();
            match crate::mtp_debug::write_mtp_dump(
                &path,
                draft_token_id,
                mtp_pos,
                base_pos,
                swap_fc_norms,
                &crate::mtp_debug::current_h_main_boundary_layers(),
                &taps,
                &extra_subops,
                &prompt_tokens,
                &replay_tokens,
                &b11_taps,
                &b12_taps,
                &c41_taps,
                &c42_taps,
                &c43_taps,
                &c44_taps,
                &c45_taps,
                &c46_taps,
                &c6_taps,
                &c7_taps,
                &c14_taps,
            ) {
                Ok(()) => tracing::info!(
                    target: "kiln::mtp_debug",
                    path = %path,
                    draft_token_id,
                    mtp_pos,
                    splice_step = ?splice_step,
                    subops = extra_subops.len(),
                    prompt_tokens_len = prompt_tokens.len(),
                    replay_tokens_len = replay_tokens.len(),
                    b11_taps = b11_taps.len(),
                    b12_taps = b12_taps.len(),
                    c41_taps = c41_taps.len(),
                    c42_taps = c42_taps.len(),
                    c43_taps = c43_taps.len(),
                    c44_taps = c44_taps.len(),
                    c45_taps = c45_taps.len(),
                    c46_taps = c46_taps.len(),
                    c6_taps = c6_taps.len(),
                    c7_taps = c7_taps.len(),
                    c14_taps = c14_taps.len(),
                    "mtp_b7_dump_written"
                ),
                Err(e) => tracing::warn!(
                    target: "kiln::mtp_debug",
                    error = %e,
                    "mtp_b7_dump_failed"
                ),
            }
        }
    } else if dump_c7_sdpa || dump_c14_post_block {
        // Defensive: drain C7 / C14 captures even when `should_dump` is false
        // to avoid leaving the TLS slots armed for the next draft step. This
        // branch should be unreachable (both flags AND with `should_dump`),
        // but is cheap insurance against future refactors that could break
        // the invariant.
        if dump_c7_sdpa {
            let _ = crate::mtp_debug::drain_c7_sdpa_capture();
        }
        if dump_c14_post_block {
            let _ = crate::mtp_debug::drain_c14_post_block_capture();
        }
    }

    // Optional Phase B instrumentation. Off by default; enabled with
    // `KILN_MTP_DEBUG=1`. See `crate::mtp_debug` for the rate-limited path.
    //
    // Phase B2 additions: halves-L2 on the `fc` input (to quantify the
    // embed-dominance hypothesis) and L2 on the fused output (to rule out
    // explode/collapse failure modes inside the fc matmul). `halves_ratio`
    // is `norm_emb_l2 / norm_h_l2`; values far from 1.0 are evidence the
    // two halves have mismatched magnitudes feeding `fc`.
    if crate::mtp_debug::should_log() {
        let h_norm = crate::mtp_debug::tensor_l2_norm(h_prev).unwrap_or(f32::NAN);
        let norm_emb_l2 = crate::mtp_debug::tensor_l2_norm(&norm_emb).unwrap_or(f32::NAN);
        let norm_h_l2 = crate::mtp_debug::tensor_l2_norm(&norm_h).unwrap_or(f32::NAN);
        let fused_l2 = crate::mtp_debug::tensor_l2_norm(&fused).unwrap_or(f32::NAN);
        let halves_ratio = if norm_h_l2 > 0.0 {
            norm_emb_l2 / norm_h_l2
        } else {
            f32::NAN
        };
        let logits_norm = crate::mtp_debug::tensor_l2_norm(&logits).unwrap_or(f32::NAN);
        let top = crate::mtp_debug::top_k_logits(&logits, 5)
            .map(|t| crate::mtp_debug::format_top_k(&t))
            .unwrap_or_else(|e| format!("<top_k err: {e}>"));
        tracing::info!(
            target: "kiln::mtp_debug",
            mtp_pos = mtp_pos,
            last_token = draft_token_id,
            swap_fc_norms = swap_fc_norms,
            h_prev_l2 = h_norm,
            norm_emb_l2 = norm_emb_l2,
            norm_h_l2 = norm_h_l2,
            halves_ratio = halves_ratio,
            fused_l2 = fused_l2,
            mtp_logits_l2 = logits_norm,
            mtp_top5 = %top,
            "mtp_draft"
        );
    }

    Ok((logits, mtp_hidden))
}

/// Controls the LM head behaviour at the end of a paged forward pass.
///
/// The streaming/tiled prefill path needs to skip the LM head entirely on
/// every non-final tile (its outputs are discarded by the caller) and
/// optionally collapse the final tile's projection to a single row, since
/// only the last token's logits feed sampling. Both shortcuts preserve
/// bit-exact agreement with the monolithic path on the values that are
/// actually consumed downstream.
#[derive(Clone, Copy, Debug)]
enum LmHeadMode {
    /// Compute the LM head over every position. Result has shape
    /// `[1, seq_len, vocab_size]`. This is the legacy `model_forward_paged`
    /// behaviour and the only mode used by training / parity verification.
    Full,
    /// Compute the LM head over the final token only. Result has shape
    /// `[1, 1, vocab_size]`. Numerically identical to slicing the last row
    /// of `Full` because RMSNorm is per-position and the matmul reduces
    /// along `hidden_size` only.
    LastRowOnly,
    /// Compute the final token's greedy argmax without materializing logits
    /// when a backend-specific fused head supports it. Used only by greedy
    /// single-token decode.
    LastRowArgmaxOnly,
    /// Compute the LM head over every position AND return the last-row
    /// pre-final-norm hidden state. Used by
    /// [`model_forward_paged_with_last_hidden`] to surface per-position logits
    /// for MTP speculative verification at position 0 (draft comparison) and
    /// position 1 (bonus), plus `h_prev` for the next MTP step.
    FullWithLastHidden,
    /// Compute the LM head over the final token only AND return the last-row
    /// pre-final-norm hidden state. Used by MTP prefill, which only consumes
    /// the next-token distribution for the prompt's final row.
    LastRowWithLastHidden,
    /// Skip RMSNorm + LM head entirely and return `None`. Used for non-final
    /// tiles where the caller throws away the logits.
    Skip,
    /// Skip RMSNorm + LM head but return the final hidden state. Used by the
    /// batched decode actor so it can project/sample rows with bounded LM-head
    /// workspace after the batch-shaped transformer pass.
    HiddenOnly,
}

/// Internal per-tile forward pass shared by `model_forward_paged` and
/// `model_forward_paged_streaming`. `lm_head_mode` controls whether the
/// final RMSNorm + LM head projection runs and over how many positions.
///
/// Pure code motion from the original `model_forward_paged` — the layer
/// loop, RoPE position tensor handling, and per-layer dispatch are unchanged.
/// The only difference is the LM head section at the bottom, which becomes
/// a `match` over `lm_head_mode`.
#[allow(clippy::too_many_arguments)]
fn model_forward_paged_inner(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    token_ids_gpu: Option<&Tensor>,
    positions_gpu: Option<&Tensor>,
    #[cfg(feature = "cuda")] graph_inputs: Option<&PagedDecodeGraphInputs<'_>>,
    lm_head_mode: LmHeadMode,
) -> Result<(Option<Tensor>, Option<Tensor>, Option<u32>)> {
    let seq_len = token_ids.len();
    let device = weights.embed_tokens.device();

    // 1. Embedding lookup: [seq_len, hidden_size]
    let mut hidden = match token_ids_gpu {
        Some(index) => embedding_lookup_from_weights_with_index(index, weights)?,
        None => embedding_lookup_from_weights(token_ids, weights)?,
    };

    // Add batch dimension: [1, seq_len, hidden_size]
    hidden = hidden.unsqueeze(0)?;

    // Phase B11b tap: `tok_embed`. Output of `embed_tokens(input_ids)` with a
    // leading batch dim. Taken once at layer 0 entry so both kiln and the HF
    // reference dump compare the exact same pre-layer hidden state. Shape
    // [1, T, hidden].
    crate::mtp_debug::capture_b11_layer0_tap("tok_embed", &hidden)?;

    // Position tensor for RoPE — use pre-allocated GPU tensor if provided,
    // otherwise create one from scratch. The pre-allocated path is essential
    // for CUDA graph replay where the tensor pointer must be stable.
    let positions_owned;
    let positions: &Tensor = match positions_gpu {
        Some(t) => t,
        None => {
            let pos_f32: Vec<f32> = (start_pos..start_pos + seq_len).map(|p| p as f32).collect();
            positions_owned = Tensor::new(pos_f32.as_slice(), device)?;
            &positions_owned
        }
    };
    let graph_rope_tables = {
        #[cfg(feature = "cuda")]
        {
            graph_inputs.map(|inputs| (inputs.rotary_cos, inputs.rotary_sin))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Option::<(&Tensor, &Tensor)>::None
        }
    };
    let rope_tables_owned = if positions_gpu.is_none() && graph_rope_tables.is_none() {
        Some(rotary_tables_from_tensor(
            positions,
            &weights.rotary_inv_freq,
        )?)
    } else {
        None
    };
    let rope_tables = graph_rope_tables.or_else(|| {
        rope_tables_owned
            .as_ref()
            .map(|(cos, sin)| (cos as &Tensor, sin as &Tensor))
    });

    // 2. Loop through all transformer layers
    let mut full_attn_idx: usize = 0;
    let mut linear_attn_idx: usize = 0;
    let profile_paged_layers = profile_paged_layers_enabled();
    let profile_gdn_stages = profile_gdn_stages_enabled();
    let profile_mlp_stages = profile_mlp_stages_enabled();
    for (i, layer) in weights.layers.iter().enumerate() {
        // Get LoRA weights for this layer, if available
        let layer_lora: Option<(&LoraLayerWeights, f32)> =
            lora.and_then(|lw| lw.layers.get(i).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                let layer_profile_start = if profile_paged_layers {
                    synchronize_for_profile(device)?;
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                // Phase B12: tell the capture layer that we are entering the
                // base-model layer `i`. `capture_b12_gqa_tap` call sites inside
                // `gqa_attention_paged` / `transformer_block_paged` gate on
                // this TLS slot + the armed capture window so that only
                // layer 31 emits sub-op taps. No-op on the production path.
                crate::mtp_debug::enter_b12_layer_scope(i);
                let block_result = transformer_block_paged_with_rope_tables(
                    backend,
                    &hidden,
                    layer,
                    config,
                    positions,
                    start_pos,
                    config.num_attention_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.rotary_dim(),
                    &weights.rotary_inv_freq,
                    rope_tables,
                    config.rms_norm_eps,
                    paged_cache,
                    block_table,
                    full_attn_idx,
                    layer_lora,
                    #[cfg(feature = "cuda")]
                    graph_inputs,
                    profile_mlp_stages.then_some((i, start_pos)),
                );
                crate::mtp_debug::exit_b12_layer_scope();
                hidden = block_result
                    .with_context(|| format!("transformer block {i} (full attention, paged)"))?;
                full_attn_idx += 1;
                if let Some(start) = layer_profile_start {
                    synchronize_for_profile(device)?;
                    log_paged_layer_profile(i, "full", seq_len, start_pos, start.elapsed());
                }
            }
            GpuAttentionWeights::Linear(lin_weights) => {
                let layer_profile_start = if profile_paged_layers {
                    synchronize_for_profile(device)?;
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let state = linear_state.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("linear attention state required for GDN layers (layer {i})")
                })?;
                let capture_b11_taps = crate::mtp_debug::should_capture_b11_tap_for_layer(i);
                let capture_c41_taps = crate::mtp_debug::should_capture_c41_layer1_tap_for_layer(i);
                let capture_c42_taps =
                    crate::mtp_debug::should_capture_c42_layer1_norm_tap_for_layer(i);
                let capture_c43_taps =
                    crate::mtp_debug::should_capture_c43_layer1_preweight_tap_for_layer(i);
                let capture_c44_taps =
                    crate::mtp_debug::should_capture_c44_layer1_f32_row_tap_for_layer(i);
                let capture_c45_taps =
                    crate::mtp_debug::should_capture_c45_layer1_row_tap_for_layer(i);
                let capture_c46_taps =
                    crate::mtp_debug::should_capture_c46_layer1_row_provenance_tap_for_layer(i);
                let use_metal_decode_ffn = seq_len == 1
                    && start_pos > 0
                    && !capture_b11_taps
                    && !capture_c41_taps
                    && !capture_c42_taps
                    && !capture_c43_taps
                    && !capture_c44_taps
                    && !capture_c45_taps
                    && !capture_c46_taps
                    && !crate::mtp_debug::is_mtp_single_token_self_attn_armed();
                if capture_c42_taps {
                    capture_c42_layer1_input_norm_taps(
                        &hidden,
                        &layer.input_layernorm,
                        config.rms_norm_eps,
                    )?;
                }
                if capture_c43_taps {
                    capture_c43_layer1_preweight_taps(
                        &hidden,
                        &layer.input_layernorm,
                        config.rms_norm_eps,
                    )?;
                }
                if capture_c44_taps {
                    capture_c44_layer1_f32_row_taps(&hidden, config.rms_norm_eps)?;
                }
                if capture_c45_taps {
                    capture_c45_layer1_row_taps(&hidden, config.rms_norm_eps)?;
                }
                if capture_c46_taps {
                    capture_c46_layer1_row_provenance_taps(&hidden)?;
                }
                let normed = {
                    kiln_nvtx::range!(c"kiln/norm/pre_attn");
                    rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?
                };
                // Phase B11b tap: `layer_0_post_input_norm`. Captured only on
                // layer 0 (the B10 scan localized divergence there) — pre-GDN
                // input LayerNorm output. Shape [1, T, hidden]. HF mirror:
                // `hidden_states` after `self.input_layernorm(...)` at layer 0.
                if capture_b11_taps {
                    crate::mtp_debug::capture_b11_layer0_tap("layer_0_post_input_norm", &normed)?;
                }
                if capture_c41_taps {
                    crate::mtp_debug::capture_c41_layer1_tap("layer_1_post_input_norm", &normed)?;
                }
                let attn_out = gated_deltanet_forward_decode_if(
                    backend,
                    &normed,
                    lin_weights,
                    config,
                    &mut state.recurrent_states[linear_attn_idx],
                    &mut state.conv_states[linear_attn_idx],
                    capture_b11_taps,
                    capture_c41_taps,
                    true,
                    use_metal_decode_ffn,
                    profile_gdn_stages.then_some((i, start_pos)),
                    true,
                    true,
                    layer_lora,
                )
                .with_context(|| format!("gated deltanet layer {i} (linear attention, paged)"))?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    (hidden + attn_out)?
                };
                if capture_c41_taps {
                    crate::mtp_debug::capture_c41_layer1_tap(
                        "layer_1_post_attn_residual",
                        &hidden,
                    )?;
                }
                let normed_post = {
                    kiln_nvtx::range!(c"kiln/norm/pre_mlp");
                    rms_norm(
                        &hidden,
                        &layer.post_attention_layernorm,
                        config.rms_norm_eps,
                    )?
                };
                let ffn_out = swiglu_ffn_backend_profiled(
                    backend,
                    &normed_post,
                    &layer.mlp,
                    layer_lora,
                    use_metal_decode_ffn,
                    profile_mlp_stages.then_some((i, start_pos)),
                )?;
                hidden = {
                    kiln_nvtx::range!(c"kiln/residual");
                    (hidden + ffn_out)?
                };
                if capture_c41_taps {
                    crate::mtp_debug::capture_c41_layer1_tap("layer_1_output", &hidden)?;
                }
                linear_attn_idx += 1;
                if let Some(start) = layer_profile_start {
                    synchronize_for_profile(device)?;
                    log_paged_layer_profile(i, "linear", seq_len, start_pos, start.elapsed());
                }
            }
        }

        // Phase B10: capture last-row hidden state at boundary layers when
        // `KILN_MTP_DUMP_HIDDEN_STATES=1` and a capture window has been armed
        // (done by `model_forward_paged_with_last_hidden`). Gate is a cheap
        // TLS-borrow + array-contains check when disarmed; zero cost in
        // production. The narrow+contiguous copies ~H floats per captured
        // layer (5 layers × 2560 f32 ≈ 50 KiB total) which is negligible
        // next to the full hidden tensor.
        if crate::mtp_debug::should_capture_hidden_state_for_layer(i) {
            let last_row = hidden.narrow(1, seq_len - 1, 1)?.contiguous()?;
            let _ = crate::mtp_debug::capture_h_main_tap(&format!("h_layer_{i}"), &last_row);
        }
    }

    // 3. Final RMSNorm + 4. LM head projection (weight-tied)
    //
    // `Full` matches the legacy code path exactly. `LastRowOnly` slices the
    // hidden tensor to the last position before the projection so we only
    // do `vocab_size * hidden_size` MACs instead of `seq_len * vocab_size *
    // hidden_size` — bit-exact with `Full`'s last row because RMSNorm is
    // per-position and the matmul reduces along `hidden_size` only. `Skip`
    // returns `None` and is used by the streaming dispatcher for every tile
    // whose logits the caller will throw away.
    match lm_head_mode {
        LmHeadMode::Full => {
            let logits = {
                kiln_nvtx::range!(c"kiln/lm_head");
                hidden = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
                lm_head_forward_backend_decode_if(Some(backend), &hidden, &weights.embed_tokens_t)?
            };
            Ok((Some(logits), None, None))
        }
        LmHeadMode::LastRowOnly => {
            let logits = {
                kiln_nvtx::range!(c"kiln/lm_head");
                let last = hidden.narrow(1, seq_len - 1, 1)?;
                let normed = rms_norm(&last, &weights.final_norm, config.rms_norm_eps)?;
                lm_head_forward_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)?
            };
            Ok((Some(logits), None, None))
        }
        LmHeadMode::LastRowArgmaxOnly => {
            let token = {
                kiln_nvtx::range!(c"kiln/lm_head_argmax");
                let last = hidden.narrow(1, seq_len - 1, 1)?;
                if let Some(token) = lm_head_weighted_prep_argmax(
                    &last,
                    &weights.final_norm,
                    &weights.embed_tokens_t,
                )? {
                    return Ok((None, None, Some(token)));
                }
                let normed = rms_norm(&last, &weights.final_norm, config.rms_norm_eps)?;
                lm_head_argmax_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)?
            };
            Ok((None, None, Some(token)))
        }
        LmHeadMode::FullWithLastHidden => {
            // Phase C18: `h_prev` must be returned POST-final-norm.
            // vLLM (`Qwen3_5MultiTokenPredictor.forward`) and SGLang consume
            // the base model's `last_hidden_state` (post-`model.norm`) as the
            // input to `pre_fc_norm_hidden`. C17 cross-referenced the upstream
            // contract and the C15 numerical fingerprint (2.0–2.4× kiln/HF
            // magnitude ratio) confirmed kiln was one RMSNorm behind. We now
            // apply `final_norm` ONCE and slice the last row from the normed
            // tensor for both the logits projection and the returned h_prev.
            let normed = {
                kiln_nvtx::range!(c"kiln/final_norm");
                rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?
            };
            let last_hidden = normed.narrow(1, seq_len - 1, 1)?.contiguous()?;
            if crate::mtp_debug::is_h_main_capture_armed() {
                let _ = crate::mtp_debug::capture_h_main_tap("h_post_final_norm", &last_hidden);
            }
            let logits = {
                kiln_nvtx::range!(c"kiln/lm_head");
                lm_head_forward_backend_decode_if(Some(backend), &normed, &weights.embed_tokens_t)?
            };
            Ok((Some(logits), Some(last_hidden), None))
        }
        LmHeadMode::LastRowWithLastHidden => {
            // Phase C18: same frame fix as `FullWithLastHidden`. For the
            // single-row variant we still only materialise the last row before
            // `final_norm` (cheap) — that row, once normed, is the canonical
            // post-final-norm `h_prev` the MTP head expects.
            let last_pre_norm = hidden.narrow(1, seq_len - 1, 1)?.contiguous()?;
            let last_hidden = {
                kiln_nvtx::range!(c"kiln/final_norm");
                rms_norm(&last_pre_norm, &weights.final_norm, config.rms_norm_eps)?
            };
            if crate::mtp_debug::is_h_main_capture_armed() {
                let _ = crate::mtp_debug::capture_h_main_tap("h_post_final_norm", &last_hidden);
            }
            let logits = {
                kiln_nvtx::range!(c"kiln/lm_head");
                lm_head_forward_backend_decode_if(
                    Some(backend),
                    &last_hidden,
                    &weights.embed_tokens_t,
                )?
            };
            Ok((Some(logits), Some(last_hidden), None))
        }
        LmHeadMode::Skip => Ok((None, None, None)),
        LmHeadMode::HiddenOnly => Ok((None, Some(hidden), None)),
    }
}

/// Streaming/tiled paged prefill — the Phase 7 long-context entry point.
///
/// Iterates `token_ids` in fixed-size tiles (default 8192 tokens, configurable
/// via `KILN_STREAMING_TILE_TOKENS`, must be a multiple of `GDN_CHUNK_SIZE`)
/// and dispatches each tile through `model_forward_paged_inner`. The
/// `LinearAttentionState` carries GDN recurrent + conv state across tile
/// boundaries; the paged KV cache is filled tile-by-tile via `start_pos +
/// cursor`. Only the final tile runs the LM head — non-final tiles use
/// `LmHeadMode::Skip`. When `KILN_STREAMING_LAST_TOKEN_LM_HEAD=0` the final
/// tile uses `LmHeadMode::Full` instead so callers can compare per-position
/// logits against the monolithic path.
///
/// Returns logits with shape `[1, 1, vocab_size]` (last-token only) or
/// `[1, last_tile_len, vocab_size]` when full LM head is requested.
///
/// `positions_gpu` is intentionally not threaded through to per-tile calls —
/// each tile builds its own per-tile position vector inside the inner fn.
/// Streaming prefill is incompatible with CUDA graph replay (which requires
/// a stable shape per call) and is only used outside of graph-captured paths.
pub fn model_forward_paged_streaming(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    model_forward_paged_streaming_with_progress(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn model_forward_paged_streaming_with_progress(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    progress: Option<&crate::cancel::CancelHandle>,
) -> Result<Tensor> {
    model_forward_paged_streaming_with(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        streaming_tile_tokens_for(weights.embed_tokens.device()),
        streaming_last_token_lm_head(),
        progress,
    )
}

/// Streaming/tiled MTP prefill.
///
/// Same tiled execution as [`model_forward_paged_streaming`], but the final
/// tile returns both last-token logits and the post-final-norm `h_prev` needed
/// to seed native MTP decoding.
pub fn model_forward_paged_streaming_last_token_with_last_hidden(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<(Tensor, Tensor)> {
    model_forward_paged_streaming_last_token_with_last_hidden_with(
        backend,
        token_ids,
        weights,
        config,
        paged_cache,
        block_table,
        start_pos,
        linear_state,
        lora,
        streaming_tile_tokens_for(weights.embed_tokens.device()),
    )
}

/// Explicit-tile variant of
/// [`model_forward_paged_streaming_last_token_with_last_hidden`].
pub fn model_forward_paged_streaming_last_token_with_last_hidden_with(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    tile_size: usize,
) -> Result<(Tensor, Tensor)> {
    let total = token_ids.len();
    if total == 0 {
        anyhow::bail!(
            "model_forward_paged_streaming_last_token_with_last_hidden requires at least one token"
        );
    }
    if tile_size == 0 || tile_size % GDN_CHUNK_SIZE != 0 {
        anyhow::bail!(
            "streaming tile_size must be a positive multiple of GDN_CHUNK_SIZE ({}), got {tile_size}",
            GDN_CHUNK_SIZE
        );
    }

    let mut last_logits: Option<Tensor> = None;
    let mut last_hidden: Option<Tensor> = None;
    let mut cursor = 0usize;
    while cursor < total {
        let end = (cursor + tile_size).min(total);
        let is_last_tile = end == total;
        let mode = if is_last_tile {
            crate::mtp_debug::arm_h_main_capture();
            crate::mtp_debug::stash_h_main_replay_context(token_ids);
            crate::mtp_debug::arm_b11_layer0_capture();
            LmHeadMode::LastRowWithLastHidden
        } else {
            LmHeadMode::Skip
        };

        let state_for_tile: Option<&mut LinearAttentionState> = linear_state.as_deref_mut();
        let (tile_logits, tile_hidden, _token) = model_forward_paged_inner(
            backend,
            &token_ids[cursor..end],
            weights,
            config,
            paged_cache,
            block_table,
            start_pos + cursor,
            state_for_tile,
            lora,
            None,
            None,
            #[cfg(feature = "cuda")]
            None,
            mode,
        )
        .with_context(|| {
            format!(
                "streaming MTP prefill tile [{cursor}, {end}) of {total} (start_pos={})",
                start_pos + cursor
            )
        })?;

        if is_last_tile {
            last_logits = tile_logits;
            last_hidden = tile_hidden;
        }

        cursor = end;
    }

    Ok((
        last_logits.context("streaming MTP prefill produced no logits")?,
        last_hidden.context("streaming MTP prefill produced no h_prev")?,
    ))
}

/// Explicit-parameter variant of [`model_forward_paged_streaming`] used by
/// tests that need to exercise specific tile sizes without manipulating
/// process-wide env vars (which would race under parallel test runners).
///
/// `tile_size` must be a positive multiple of `GDN_CHUNK_SIZE`.
pub fn model_forward_paged_streaming_with(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    tile_size: usize,
    last_token_only: bool,
    progress: Option<&crate::cancel::CancelHandle>,
) -> Result<Tensor> {
    let total = token_ids.len();
    if total == 0 {
        anyhow::bail!("model_forward_paged_streaming requires at least one token");
    }
    if tile_size == 0 || tile_size % GDN_CHUNK_SIZE != 0 {
        anyhow::bail!(
            "streaming tile_size must be a positive multiple of GDN_CHUNK_SIZE ({}), got {tile_size}",
            GDN_CHUNK_SIZE
        );
    }

    let mut last_logits: Option<Tensor> = None;
    let mut cursor = 0usize;
    while cursor < total {
        let end = (cursor + tile_size).min(total);
        let is_last_tile = end == total;
        let mode = if is_last_tile {
            if last_token_only {
                LmHeadMode::LastRowOnly
            } else {
                LmHeadMode::Full
            }
        } else {
            LmHeadMode::Skip
        };

        // Re-borrow the optional `&mut LinearAttentionState` for this tile.
        // `Option<&mut T>::as_deref_mut()` produces `Option<&mut T>` again.
        let state_for_tile: Option<&mut LinearAttentionState> = linear_state.as_deref_mut();

        let (tile_logits, _tile_hidden, _token) = model_forward_paged_inner(
            backend,
            &token_ids[cursor..end],
            weights,
            config,
            paged_cache,
            block_table,
            start_pos + cursor,
            state_for_tile,
            lora,
            None,
            None,
            #[cfg(feature = "cuda")]
            None,
            mode,
        )
        .with_context(|| {
            format!(
                "streaming prefill tile [{cursor}, {end}) of {total} (start_pos={})",
                start_pos + cursor
            )
        })?;

        if is_last_tile {
            last_logits = tile_logits;
        }

        cursor = end;
        if let Some(progress) = progress {
            progress.report_prefill_tokens_completed(cursor as u64);
        }
    }

    last_logits.context("streaming prefill produced no logits (empty token_ids)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use candle_core::Var;

    /// Module-local mutex for tests that mutate process-wide env vars
    /// (residency kill-switches, projection drop overrides). Serialises
    /// those tests against each other so `nextest`'s parallel execution
    /// doesn't observe a half-mutated environment.
    ///
    /// Module-local because `kiln_core::env_flag::TEST_ENV_LOCK` is
    /// `cfg(test)`-gated and only visible inside kiln-core's own test
    /// build — from another crate's tests it appears unresolved.
    static RESIDENCY_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Regression test for the 2026-05-12 → 2026-05-14 silent inference
    /// outage. Commit 997a608f widened `drop_projection_transposes_enabled`
    /// from "training is engaged" to "Vulkan is the active backend OR
    /// training is engaged" — which silently replaced every projection
    /// transpose tensor (`in_proj_qkv_t`, `in_proj_z_t`, `out_proj_t`,
    /// `q_proj_t`, etc.) with `Tensor::zeros((1,), DType::BF16, ...)` at
    /// load time. Inference reads those caches directly via
    /// `backend.linear_prefill_apply`, and the GDN prefill kernel then
    /// bailed out with `only 2d matrixes are supported [1, T, hidden] [1]`
    /// on every single /v1/chat/completions request. The fix narrowed the
    /// gate back to "training is engaged" (KILN_VK_NATIVE_TRAINING set);
    /// `keep_projection_originals_enabled()` stays Vulkan-aware because
    /// the trainer needs the originals later.
    ///
    /// This test pins the contract: turning Vulkan on must NOT drop
    /// transposes by itself.
    #[test]
    fn vulkan_active_alone_does_not_drop_projection_transposes() {
        let _guard = RESIDENCY_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: env mutation serialized by RESIDENCY_ENV_LOCK.
        unsafe {
            std::env::remove_var("KILN_VK_NATIVE_TRAINING");
            std::env::remove_var("KILN_KEEP_PROJECTION_TRANSPOSES");
        }
        let _vk = crate::backend::test_only_set_vulkan_active(true);
        assert!(
            !drop_projection_transposes_enabled(),
            "drop_projection_transposes_enabled() must NOT return true just \
             because Vulkan is active — that breaks every chat completion on \
             Vulkan with `only 2d matrixes are supported [..., hidden] [1]`. \
             Only KILN_VK_NATIVE_TRAINING should opt in to dropping transposes."
        );
        // Sanity: enabling training mode flips it on.
        // SAFETY: env mutation serialized by RESIDENCY_ENV_LOCK.
        unsafe { std::env::set_var("KILN_VK_NATIVE_TRAINING", "1") };
        assert!(
            drop_projection_transposes_enabled(),
            "KILN_VK_NATIVE_TRAINING=1 should still enable transpose drop"
        );
        // SAFETY: env cleanup serialized by RESIDENCY_ENV_LOCK.
        unsafe { std::env::remove_var("KILN_VK_NATIVE_TRAINING") };
    }

    /// Tests all run on `Device::Cpu`, so the `CpuBackend` (all kernel methods
    /// return `Ok(None)`) is the right dispatch target.
    fn test_backend(device: &Device) -> CpuBackend {
        CpuBackend::new(device.clone())
    }

    #[derive(Debug)]
    struct FixedLinearBackend {
        device: Device,
        values: Vec<f32>,
        dims: (usize, usize, usize),
    }

    impl BackendRuntime for FixedLinearBackend {
        fn name(&self) -> &'static str {
            "fixed-linear-test"
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn linear_decode(&self, _x: &Tensor, _weight_t: &Tensor) -> Result<Option<Tensor>> {
            Ok(Some(Tensor::from_vec(
                self.values.clone(),
                self.dims,
                &self.device,
            )?))
        }
    }

    #[derive(Debug)]
    struct FixedMlpBackend {
        device: Device,
        fused_values: Option<Vec<f32>>,
        fused_dims: (usize, usize, usize),
        gate_up_values: Option<Vec<f32>>,
        gate_up_dims: (usize, usize, usize),
        fused_calls: std::sync::atomic::AtomicUsize,
        gate_up_calls: std::sync::atomic::AtomicUsize,
    }

    impl BackendRuntime for FixedMlpBackend {
        fn name(&self) -> &'static str {
            "fixed-mlp-test"
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn mlp_decode(
            &self,
            _x: &Tensor,
            _gate_weight_t: &Tensor,
            _up_weight_t: &Tensor,
            _down_weight_t: &Tensor,
        ) -> Result<Option<Tensor>> {
            self.fused_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(match self.fused_values.as_ref() {
                Some(values) => Some(Tensor::from_vec(
                    values.clone(),
                    self.fused_dims,
                    &self.device,
                )?),
                None => None,
            })
        }

        fn mlp_gate_up_decode(
            &self,
            _x: &Tensor,
            _gate_weight_t: &Tensor,
            _up_weight_t: &Tensor,
        ) -> Result<Option<Tensor>> {
            self.gate_up_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(match self.gate_up_values.as_ref() {
                Some(values) => Some(Tensor::from_vec(
                    values.clone(),
                    self.gate_up_dims,
                    &self.device,
                )?),
                None => None,
            })
        }
    }

    #[test]
    fn test_backend_linear_decode_adds_lora_delta() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 2), &device)?;
        let weight_t = Tensor::zeros((2, 3), DType::F32, &device)?;
        let lora = LoraProjectionWeights {
            a: Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &device)?,
            b: Tensor::from_vec(vec![5.0f32, 6.0, 7.0], (3, 1), &device)?,
        };
        let backend = FixedLinearBackend {
            device: device.clone(),
            values: vec![10.0, 20.0, 30.0],
            dims: (1, 1, 3),
        };

        let out = linear_with_lora_t_backend_decode_if(
            Some(&backend),
            false,
            &x,
            &weight_t,
            Some(&lora),
            0.5,
        )?;

        let values = out.flatten_all()?.to_vec1::<f32>()?;
        let expected = [37.5, 53.0, 68.5];
        for (got, expected) in values.iter().zip(expected) {
            assert!(
                (got - expected).abs() < 1e-6,
                "got {got}, expected {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_swiglu_down_only_lora_keeps_backend_gate_up_decode() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 2), &device)?;
        let zero_proj = Tensor::zeros((2, 2), DType::F32, &device)?;
        let zero_proj_t = zero_proj.t()?.contiguous()?;
        let mlp = GpuFfnWeights {
            gate_proj: zero_proj.clone(),
            up_proj: zero_proj.clone(),
            down_proj: zero_proj.clone(),
            gate_proj_t: zero_proj_t.clone(),
            up_proj_t: zero_proj_t.clone(),
            down_proj_t: zero_proj_t,
            gate_proj_marlin: None,
            up_proj_marlin: None,
            down_proj_marlin: None,
        };
        let backend = FixedMlpBackend {
            device: device.clone(),
            fused_values: None,
            fused_dims: (1, 1, 2),
            gate_up_values: Some(vec![3.0, 5.0]),
            gate_up_dims: (1, 1, 2),
            fused_calls: std::sync::atomic::AtomicUsize::new(0),
            gate_up_calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let lora_layer = LoraLayerWeights {
            down_proj: Some(LoraProjectionWeights {
                a: Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &device)?,
                b: Tensor::from_vec(vec![2.0f32, 4.0], (2, 1), &device)?,
            }),
            ..Default::default()
        };

        let out = swiglu_ffn_impl(
            Some(&backend),
            &x,
            &mlp,
            Some((&lora_layer, 1.0)),
            false,
            None,
        )?;

        assert_eq!(
            backend
                .gate_up_calls
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            backend
                .fused_calls
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        let values = out.flatten_all()?.to_vec1::<f32>()?;
        let expected = [6.0, 12.0];
        for (got, expected) in values.iter().zip(expected) {
            assert!(
                (got - expected).abs() < 1e-6,
                "got {got}, expected {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_swiglu_attention_only_lora_keeps_backend_mlp_decode() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 2), &device)?;
        let zero_proj = Tensor::zeros((2, 2), DType::F32, &device)?;
        let zero_proj_t = zero_proj.t()?.contiguous()?;
        let mlp = GpuFfnWeights {
            gate_proj: zero_proj.clone(),
            up_proj: zero_proj.clone(),
            down_proj: zero_proj.clone(),
            gate_proj_t: zero_proj_t.clone(),
            up_proj_t: zero_proj_t.clone(),
            down_proj_t: zero_proj_t,
            gate_proj_marlin: None,
            up_proj_marlin: None,
            down_proj_marlin: None,
        };
        let backend = FixedMlpBackend {
            device: device.clone(),
            fused_values: Some(vec![7.0, 11.0]),
            fused_dims: (1, 1, 2),
            gate_up_values: Some(vec![3.0, 5.0]),
            gate_up_dims: (1, 1, 2),
            fused_calls: std::sync::atomic::AtomicUsize::new(0),
            gate_up_calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let lora_layer = LoraLayerWeights {
            q_proj: Some(LoraProjectionWeights {
                a: Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &device)?,
                b: Tensor::from_vec(vec![2.0f32, 4.0], (2, 1), &device)?,
            }),
            ..Default::default()
        };

        let out = swiglu_ffn_impl(
            Some(&backend),
            &x,
            &mlp,
            Some((&lora_layer, 1.0)),
            false,
            None,
        )?;

        assert_eq!(
            backend
                .fused_calls
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            backend
                .gate_up_calls
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        let values = out.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(values, vec![7.0, 11.0]);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_linear_decode_lora_matches_broadcast_matmul() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!(
                "Metal unavailable, skipping test_metal_linear_decode_lora_matches_broadcast_matmul"
            );
            return Ok(());
        };

        let input_dim = 128usize;
        let output_dim = 133usize;
        let mut exercised_fast_path = false;
        for rank in [4usize, 32usize, 64usize] {
            for batch in [1usize, 4usize] {
                let x = patterned_bf16(&[batch, 1usize, input_dim], 0.01, &device)?;
                let weight_t = patterned_bf16(&[input_dim, output_dim], 0.0078125, &device)?;
                let lora = LoraProjectionWeights {
                    a: patterned_bf16(&[rank, input_dim], 0.001, &device)?,
                    b: patterned_bf16(&[output_dim, rank], 0.0015, &device)?,
                };
                let supported = if batch == 1 {
                    crate::backend::metal::metal_transposed_coop_gemv_supports(&x, &weight_t)
                } else {
                    crate::backend::metal::metal_transposed_coop_gemv_decode_batch_supports(
                        &x, &weight_t,
                    )
                };
                if !supported {
                    eprintln!(
                        "Metal transposed coop GEMV disabled for rank={rank} batch={batch}, skipping LoRA parity row"
                    );
                    continue;
                }
                exercised_fast_path = true;

                let fallback = linear_with_lora_t(&x, &weight_t, Some(&lora), 0.75)?;
                let fast = linear_with_lora_t_decode(&x, &weight_t, Some(&lora), 0.75)?;

                assert_eq!(fast.dims(), &[batch, 1usize, output_dim]);
                assert_eq!(fast.dtype(), DType::BF16);

                let (max, mean) = tensor_abs_diff_stats(&fallback, &fast)?;
                assert!(
                    max < 2e-2,
                    "Metal LoRA linear decode rank={rank} batch={batch} max_abs_diff={max:e} exceeds tolerance"
                );
                assert!(
                    mean < 3e-3,
                    "Metal LoRA linear decode rank={rank} batch={batch} mean_abs_diff={mean:e} exceeds tolerance"
                );
            }
        }

        if !exercised_fast_path {
            eprintln!("Metal transposed coop GEMV unavailable, no LoRA fast path rows exercised");
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "synthetic Metal LoRA projection microbench; run explicitly with --ignored --nocapture"]
    fn bench_metal_linear_decode_lora_qwen35_synthetic() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };
        let warmup = std::env::var("KILN_METAL_LORA_LINEAR_BENCH_WARMUP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2);
        let iters = std::env::var("KILN_METAL_LORA_LINEAR_BENCH_ITERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(5);

        for rank in [1usize, 2usize, 4usize, 8usize, 16usize, 32usize, 64usize] {
            for batch in [1usize, 2usize, 4usize, 8usize] {
                bench_metal_lora_linear_case(
                    &device,
                    "mlp_gate_or_up",
                    batch,
                    2560,
                    9216,
                    rank,
                    warmup,
                    iters,
                )?;
                bench_metal_lora_linear_case(
                    &device,
                    "down_proj",
                    batch,
                    9216,
                    2560,
                    rank,
                    warmup,
                    iters,
                )?;
            }
        }

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "synthetic Metal QKV-shaped LoRA projection microbench; run explicitly with --ignored --nocapture"]
    fn bench_metal_linear_decode_lora_qwen35_qkv_synthetic() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };
        let warmup = std::env::var("KILN_METAL_LORA_QKV_LINEAR_BENCH_WARMUP")
            .or_else(|_| std::env::var("KILN_METAL_LORA_LINEAR_BENCH_WARMUP"))
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2);
        let iters = std::env::var("KILN_METAL_LORA_QKV_LINEAR_BENCH_ITERS")
            .or_else(|_| std::env::var("KILN_METAL_LORA_LINEAR_BENCH_ITERS"))
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(5);

        for rank in [1usize, 2usize, 4usize, 8usize, 16usize, 32usize, 64usize] {
            for batch in [1usize, 2usize, 4usize, 8usize] {
                bench_metal_lora_linear_case(
                    &device, "q_proj", batch, 2560, 8192, rank, warmup, iters,
                )?;
                bench_metal_lora_linear_case(
                    &device,
                    "k_or_v_proj",
                    batch,
                    2560,
                    1024,
                    rank,
                    warmup,
                    iters,
                )?;
            }
        }

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_arguments)]
    fn bench_metal_lora_linear_case(
        device: &Device,
        label: &str,
        batch: usize,
        input_dim: usize,
        output_dim: usize,
        rank: usize,
        warmup: usize,
        iters: usize,
    ) -> Result<()> {
        let x = patterned_bf16(&[batch, 1usize, input_dim], 0.01, device)?;
        let weight_t = patterned_bf16(&[input_dim, output_dim], 0.0001, device)?;
        let lora = LoraProjectionWeights {
            a: patterned_bf16(&[rank, input_dim], 0.0002, device)?,
            b: patterned_bf16(&[output_dim, rank], 0.0002, device)?,
        };
        let supported = if batch == 1 {
            crate::backend::metal::metal_transposed_coop_gemv_supports(&x, &weight_t)
        } else {
            crate::backend::metal::metal_transposed_coop_gemv_decode_batch_supports(&x, &weight_t)
        };
        if !supported {
            eprintln!("metal_lora_linear_bench label={label} skipped unsupported shape");
            return Ok(());
        }

        let fallback = linear_with_lora_t(&x, &weight_t, Some(&lora), 1.0)?;
        let fast = linear_with_lora_t_decode(&x, &weight_t, Some(&lora), 1.0)?;
        let (max, mean) = tensor_abs_diff_stats(&fallback, &fast)?;

        for _ in 0..warmup {
            let out = linear_with_lora_t_decode(&x, &weight_t, Some(&lora), 1.0)?;
            std::hint::black_box(out);
            let out = linear_with_lora_t(&x, &weight_t, Some(&lora), 1.0)?;
            std::hint::black_box(out);
        }
        synchronize_for_profile(device)?;

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let out = linear_with_lora_t_decode(&x, &weight_t, Some(&lora), 1.0)?;
            std::hint::black_box(out);
        }
        synchronize_for_profile(device)?;
        let fast_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let out = linear_with_lora_t(&x, &weight_t, Some(&lora), 1.0)?;
            std::hint::black_box(out);
        }
        synchronize_for_profile(device)?;
        let fallback_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        eprintln!(
            "metal_lora_linear_bench label={label} batch={batch} input_dim={input_dim} output_dim={output_dim} rank={rank} iters={iters} fast_ms={fast_ms:.3} fallback_ms={fallback_ms:.3} speedup={:.3} max_abs_diff={max:e} mean_abs_diff={mean:e}",
            fallback_ms / fast_ms
        );
        Ok(())
    }

    /// `stub_embed_tokens_after_upload` must fire on Metal and on
    /// Vulkan-active processes — both backends route the embedding
    /// lookup through `embed_tokens_t` and never read the raw
    /// `embed_tokens` table again, so the candle CPU mirror is
    /// pure overhead.
    ///
    /// Phase 1.2 sub-step 1: keep this contract under test so a future
    /// edit can't silently drop the Vulkan branch and reintroduce the
    /// duplicate embedding-table footprint.
    ///
    /// We deliberately do NOT call `mark_vulkan_active()` here even
    /// though it would let us assert the post-flag behavior: the flag
    /// is process-global (and `vulkan_active()` is read by other
    /// modules including the transposed weight cache writer's
    /// scheduling envelope), so flipping it inside one unit test
    /// destabilizes every later test in the same nextest process.
    /// The flag's read is a one-line public API; the integration
    /// behavior is exercised by the live-server validation in
    /// `kiln-server`.
    #[test]
    fn test_stub_embed_tokens_decision_negative_only() {
        let cpu = Device::Cpu;
        // Pre-flag baseline: plain CPU, no Vulkan, must NOT stub.
        // (If a prior test in the same process leaked vulkan_active=true,
        // skip the assertion rather than make a false negative claim.)
        if !crate::backend::vulkan_active() {
            assert!(
                !stub_embed_tokens_after_upload(&cpu),
                "plain CPU with no Vulkan must NOT stub"
            );
        }
        // Cuda device path is gated by feature; rely on the predicate's
        // pattern match returning false for Device::Cpu under non-Metal
        // builds, which is what the negative assertion above covers.
    }

    /// `marlin_bf16_drop_disabled()` must default to `false` (i.e.,
    /// drop *enabled*) — that's the contract on the Vulkan training
    /// path. The kill-switch `KILN_DISABLE_MARLIN_BF16_DROP=1` is the
    /// only thing that should re-enable the duplicate BF16 residency.
    ///
    /// Pins the residency-audit claim that Marlin-absorbed BF16
    /// weights are stubbed by default. A regression that flips the
    /// default would silently double base-model footprint on the
    /// candle CPU side.
    #[test]
    fn test_marlin_bf16_drop_default_is_enabled() {
        let _guard = RESIDENCY_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Save & clear the env var so the test reads the pure default.
        let prior = std::env::var("KILN_DISABLE_MARLIN_BF16_DROP").ok();
        unsafe {
            std::env::remove_var("KILN_DISABLE_MARLIN_BF16_DROP");
        }
        let result = marlin_bf16_drop_disabled();
        // Restore the prior env state for any later test in this process.
        if let Some(prev) = prior {
            unsafe {
                std::env::set_var("KILN_DISABLE_MARLIN_BF16_DROP", prev);
            }
        }
        assert!(!result, "marlin BF16 drop must be enabled by default");
    }

    /// Kill-switch must actually fire — `KILN_DISABLE_MARLIN_BF16_DROP=1`
    /// disables the drop. Exercises the parsing logic so a typo in the
    /// matcher (e.g. lower-casing missing) would be caught.
    #[test]
    fn test_marlin_bf16_drop_kill_switch_fires() {
        let _guard = RESIDENCY_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior = std::env::var("KILN_DISABLE_MARLIN_BF16_DROP").ok();
        for value in &["1", "true", "TRUE", "yes", "Yes"] {
            unsafe {
                std::env::set_var("KILN_DISABLE_MARLIN_BF16_DROP", value);
            }
            assert!(
                marlin_bf16_drop_disabled(),
                "KILN_DISABLE_MARLIN_BF16_DROP={value} must disable the drop"
            );
        }
        if let Some(prev) = prior {
            unsafe {
                std::env::set_var("KILN_DISABLE_MARLIN_BF16_DROP", prev);
            }
        } else {
            unsafe {
                std::env::remove_var("KILN_DISABLE_MARLIN_BF16_DROP");
            }
        }
    }

    /// `keep_projection_originals_enabled()` must default to `false`
    /// (i.e., projection originals are *eligible* for the drop on the
    /// devices that ask for it). Pins the residency-audit claim that
    /// per-layer projection originals are stubbed by default on Vulkan.
    #[test]
    fn test_keep_projection_originals_default_off() {
        let _guard = RESIDENCY_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior = std::env::var("KILN_KEEP_PROJECTION_ORIGINALS").ok();
        unsafe {
            std::env::remove_var("KILN_KEEP_PROJECTION_ORIGINALS");
        }
        let result = keep_projection_originals_enabled();
        if let Some(prev) = prior {
            unsafe {
                std::env::set_var("KILN_KEEP_PROJECTION_ORIGINALS", prev);
            }
        }
        assert!(
            !result,
            "KILN_KEEP_PROJECTION_ORIGINALS must default to off (drop allowed)"
        );
    }

    /// `KILN_KEEP_PROJECTION_ORIGINALS=1` must override the default
    /// and keep the originals resident — required for A/B parity
    /// debugging.
    #[test]
    fn test_keep_projection_originals_kill_switch_fires() {
        let _guard = RESIDENCY_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior = std::env::var("KILN_KEEP_PROJECTION_ORIGINALS").ok();
        for value in &["1", "true", "yes"] {
            unsafe {
                std::env::set_var("KILN_KEEP_PROJECTION_ORIGINALS", value);
            }
            assert!(
                keep_projection_originals_enabled(),
                "KILN_KEEP_PROJECTION_ORIGINALS={value} must keep the originals"
            );
        }
        if let Some(prev) = prior {
            unsafe {
                std::env::set_var("KILN_KEEP_PROJECTION_ORIGINALS", prev);
            }
        } else {
            unsafe {
                std::env::remove_var("KILN_KEEP_PROJECTION_ORIGINALS");
            }
        }
    }

    /// `projection_original_drop_enabled_for_device(Device::Cpu)` is
    /// `false` on plain CPU absent any overrides. (The Vulkan-active
    /// and KILN_DROP_PROJECTION_ORIGINALS branches flip it true; both
    /// are exercised by the integration runs, but the predicate's CPU
    /// baseline is what this test pins.)
    #[test]
    fn test_projection_drop_cpu_default_off() {
        let _guard = RESIDENCY_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior_keep = std::env::var("KILN_KEEP_PROJECTION_ORIGINALS").ok();
        let prior_drop = std::env::var("KILN_DROP_PROJECTION_ORIGINALS").ok();
        unsafe {
            std::env::remove_var("KILN_KEEP_PROJECTION_ORIGINALS");
            std::env::remove_var("KILN_DROP_PROJECTION_ORIGINALS");
        }
        let result = if !crate::backend::vulkan_active() {
            // Safe to assert: vulkan_active=false makes the device
            // pattern-match the only deciding factor for Device::Cpu.
            Some(projection_original_drop_enabled_for_device(&Device::Cpu))
        } else {
            None
        };
        if let Some(prev) = prior_keep {
            unsafe {
                std::env::set_var("KILN_KEEP_PROJECTION_ORIGINALS", prev);
            }
        }
        if let Some(prev) = prior_drop {
            unsafe {
                std::env::set_var("KILN_DROP_PROJECTION_ORIGINALS", prev);
            }
        }
        if let Some(res) = result {
            assert!(
                !res,
                "plain CPU with no overrides must NOT drop projection originals"
            );
        }
    }

    /// Property: when `embed_tokens` is a 1-element stub (the only
    /// case `stub_embed_tokens_after_upload` produces), the dispatch in
    /// `embedding_lookup_from_weights` must route to
    /// `embedding_lookup_from_transposed`. We can't trivially build a
    /// full `GpuWeights` here, so test the dim-mismatch branch directly
    /// by checking that `dropped_weight_stub` produces a tensor whose
    /// dims will not equal `[t_dims[1], t_dims[0]]` for any non-degenerate
    /// transposed shape.
    #[test]
    fn test_dropped_stub_never_matches_real_embedding_dims() -> Result<()> {
        let device = Device::Cpu;
        let w = WeightTensor {
            dtype: crate::weights::TensorDType::F32,
            shape: vec![5, 3], // vocab=5, hidden=3
            data: crate::weights::WeightData::owned(vec![0u8; 5 * 3 * 4]),
            source: None,
        };
        let stub = dropped_weight_stub(&w, &device)?;
        let materialized_t_dims = [3usize, 5usize];
        let expected_embed_dims = [materialized_t_dims[1], materialized_t_dims[0]];
        assert_ne!(stub.dims(), expected_embed_dims.as_slice());
        assert_eq!(stub.dims(), &[1usize]);
        assert_eq!(stub.dtype(), candle_core::DType::F32);
        Ok(())
    }

    #[test]
    fn test_embedding_lookup() -> Result<()> {
        let device = Device::Cpu;
        // vocab_size=5, hidden_size=3
        let embed_data: Vec<f32> = vec![
            0.1, 0.2, 0.3, // token 0
            0.4, 0.5, 0.6, // token 1
            0.7, 0.8, 0.9, // token 2
            1.0, 1.1, 1.2, // token 3
            1.3, 1.4, 1.5, // token 4
        ];
        let embed = Tensor::new(embed_data, &device)?.reshape((5, 3))?;

        let result = embedding_lookup(&[2, 0, 4], &embed)?;
        assert_eq!(result.dims(), &[3, 3]); // [seq_len=3, hidden_size=3]

        let vals = result.to_vec2::<f32>()?;
        // Token 2
        assert!((vals[0][0] - 0.7).abs() < 1e-6);
        assert!((vals[0][1] - 0.8).abs() < 1e-6);
        assert!((vals[0][2] - 0.9).abs() < 1e-6);
        // Token 0
        assert!((vals[1][0] - 0.1).abs() < 1e-6);
        // Token 4
        assert!((vals[2][0] - 1.3).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_embedding_lookup_from_transposed_matches_table() -> Result<()> {
        let device = Device::Cpu;
        let embed_data: Vec<f32> = vec![
            0.1, 0.2, 0.3, //
            0.4, 0.5, 0.6, //
            0.7, 0.8, 0.9, //
            1.0, 1.1, 1.2, //
            1.3, 1.4, 1.5,
        ];
        let embed = Tensor::new(embed_data, &device)?.reshape((5, 3))?;
        let embed_t = embed.t()?.contiguous()?;

        let direct = embedding_lookup(&[2, 0, 4], &embed)?;
        let transposed = embedding_lookup_from_transposed(&[2, 0, 4], &embed_t)?;

        assert_eq!(transposed.dims(), direct.dims());
        assert_eq!(transposed.to_vec2::<f32>()?, direct.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn test_rms_norm_known_values() -> Result<()> {
        let device = Device::Cpu;
        // x = [1, 2, 3], weight = [0, 0, 0], eps = 0
        // Effective weight = 1 + w = [1, 1, 1]
        // RMS = sqrt(mean([1,4,9])) = sqrt(14/3) ≈ 2.1602
        // normed = [1/2.1602, 2/2.1602, 3/2.1602] ≈ [0.4629, 0.9258, 1.3887]
        let x = Tensor::new(&[1.0_f32, 2.0, 3.0], &device)?.unsqueeze(0)?; // [1, 3]
        let w = Tensor::new(&[0.0_f32, 0.0, 0.0], &device)?;

        let result = rms_norm(&x, &w, 1e-8)?;
        let vals = result.to_vec2::<f32>()?;

        let rms = (14.0_f64 / 3.0).sqrt();
        assert!((vals[0][0] as f64 - 1.0 / rms).abs() < 1e-4);
        assert!((vals[0][1] as f64 - 2.0 / rms).abs() < 1e-4);
        assert!((vals[0][2] as f64 - 3.0 / rms).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn test_rms_norm_with_weight() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[2.0_f32, 2.0, 2.0], &device)?.unsqueeze(0)?;
        let w = Tensor::new(&[0.5_f32, 1.0, 2.0], &device)?;

        let result = rms_norm(&x, &w, 1e-8)?;
        let vals = result.to_vec2::<f32>()?;

        // RMS of [2,2,2] = 2.0, so normed = [1,1,1]
        // Effective weight = 1 + w = [1.5, 2.0, 3.0]
        // After weight: [1.5, 2.0, 3.0]
        assert!((vals[0][0] - 1.5).abs() < 1e-4);
        assert!((vals[0][1] - 2.0).abs() < 1e-4);
        assert!((vals[0][2] - 3.0).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn test_c45_row_replay_matches_production_broadcast_mul_last_row() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2usize;
        let seq_len = 3usize;
        let hidden = 4usize;
        let eps = 1e-6;
        let x = Tensor::from_slice(
            &[
                1.0_f32, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5, 4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0,
                -4.0, 1.0, -1.5, 2.0, -2.5, 0.25, -0.5, 0.75, -1.0,
            ],
            (batch, seq_len, hidden),
            &device,
        )?;

        let (
            rms_inv_row,
            extracted_scalars,
            last_row_values,
            broadcast_output,
            scalar_values,
            reconstructed,
        ) = c45_layer1_row_replay_tensors(&x, eps)?;

        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rms_inv = (variance + eps)?.sqrt()?.recip()?;
        let production = x_f32.broadcast_mul(&rms_inv)?;
        let production_last_row = production.narrow(1, seq_len - 1, 1)?.contiguous()?;
        let production_last_row_values = x_f32
            .narrow(1, seq_len - 1, 1)?
            .contiguous()?
            .reshape((batch, hidden))?
            .contiguous()?;
        let production_scalar_values =
            production_last_row.reshape((batch, hidden))?.contiguous()?;
        let production_rms_inv_row = rms_inv.narrow(1, seq_len - 1, 1)?.contiguous()?;

        assert_eq!(
            rms_inv_row.to_vec3::<f32>()?,
            production_rms_inv_row.to_vec3::<f32>()?
        );
        assert_eq!(
            extracted_scalars.to_vec1::<f32>()?,
            production_rms_inv_row.reshape((batch,))?.to_vec1::<f32>()?
        );
        assert_eq!(
            last_row_values.to_vec2::<f32>()?,
            production_last_row_values.to_vec2::<f32>()?
        );
        assert_eq!(
            broadcast_output.to_vec3::<f32>()?,
            production_last_row.to_vec3::<f32>()?
        );
        assert_eq!(
            reconstructed.to_vec3::<f32>()?,
            production_last_row.to_vec3::<f32>()?
        );
        assert_eq!(
            scalar_values.to_vec2::<f32>()?,
            production_scalar_values.to_vec2::<f32>()?
        );
        assert_eq!(
            reconstructed.reshape((batch, hidden))?.to_vec2::<f32>()?,
            scalar_values.to_vec2::<f32>()?
        );
        assert_eq!(
            broadcast_output
                .reshape((batch, hidden))?
                .to_vec2::<f32>()?,
            scalar_values.to_vec2::<f32>()?
        );

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_gdn_gated_rms_norm_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!(
                    "CUDA unavailable, skipping test_cuda_gdn_gated_rms_norm_matches_fallback: {err}"
                );
                return Ok(());
            }
        };
        let backend = crate::backend::for_device(&device);
        if !backend.supports_gdn_gated_rms_norm() {
            eprintln!("CUDA gated RMSNorm disabled, skipping parity test");
            return Ok(());
        }

        let batch = 1usize;
        let seq_len = 3usize;
        let heads = 32usize;
        let hidden = 128usize;
        let elems = batch * seq_len * heads * hidden;

        let mut rng = StdRng::seed_from_u64(0xC0DA_6A7E);
        let x_data: Vec<f32> = (0..elems)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let z_data: Vec<f32> = (0..elems)
            .map(|_| rng.random_range(-2.0f32..2.0f32))
            .collect();
        let w_data: Vec<f32> = (0..hidden)
            .map(|_| rng.random_range(0.5f32..1.5f32))
            .collect();

        let x = Tensor::from_slice(&x_data, (batch, seq_len, heads, hidden), &device)?
            .to_dtype(DType::BF16)?;
        let z = Tensor::from_slice(&z_data, (batch, seq_len, heads, hidden), &device)?
            .to_dtype(DType::BF16)?;
        let weight = Tensor::from_slice(&w_data, (hidden,), &device)?.to_dtype(DType::BF16)?;

        let fallback = gated_rms_norm_fallback(&x, &z, &weight, 1e-6)?;
        let fused = backend
            .gdn_gated_rms_norm(&x, &z, &weight, 1e-6)?
            .context("CUDA backend declined gated RMSNorm test shape")?;

        assert_eq!(fused.dims(), fallback.dims());
        assert_eq!(fused.dtype(), DType::BF16);

        let diff = (fused.to_dtype(DType::F32)?
            - fallback.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!("gated_rms_norm cuda vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
        assert!(
            max < 5e-3,
            "CUDA gated_rms_norm max_abs_diff={max:e} exceeds 5e-3"
        );
        assert!(
            mean < 5e-4,
            "CUDA gated_rms_norm mean_abs_diff={mean:e} exceeds 5e-4"
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_gated_rms_norm_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!("Metal unavailable, skipping test_metal_gated_rms_norm_matches_fallback");
            return Ok(());
        };
        let backend = crate::backend::for_device(&device);
        if !backend.supports_gdn_gated_rms_norm() {
            eprintln!("Metal gated RMSNorm disabled, skipping parity test");
            return Ok(());
        }

        let batch = 1usize;
        let seq_len = 3usize;
        let heads = 32usize;
        let hidden = 128usize;
        let elems = batch * seq_len * heads * hidden;

        let mut rng = StdRng::seed_from_u64(0x6A7E_DA75);
        let x_data: Vec<f32> = (0..elems)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let z_data: Vec<f32> = (0..elems)
            .map(|_| rng.random_range(-2.0f32..2.0f32))
            .collect();
        let w_data: Vec<f32> = (0..hidden)
            .map(|_| rng.random_range(0.5f32..1.5f32))
            .collect();

        let x = Tensor::from_slice(&x_data, (batch, seq_len, heads, hidden), &device)?
            .to_dtype(DType::BF16)?;
        let z = Tensor::from_slice(&z_data, (batch, seq_len, heads, hidden), &device)?
            .to_dtype(DType::BF16)?;
        let weight = Tensor::from_slice(&w_data, (hidden,), &device)?.to_dtype(DType::BF16)?;

        let fallback = gated_rms_norm_fallback(&x, &z, &weight, 1e-6)?;
        let fused = backend
            .gdn_gated_rms_norm(&x, &z, &weight, 1e-6)?
            .context("Metal backend declined gated RMSNorm test shape")?;

        assert_eq!(fused.dims(), fallback.dims());
        assert_eq!(fused.dtype(), DType::BF16);

        let diff = (fused.to_dtype(DType::F32)?
            - fallback.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!("gated_rms_norm metal vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
        assert!(
            max < 5e-3,
            "Metal gated_rms_norm max_abs_diff={max:e} exceeds 5e-3"
        );
        assert!(
            mean < 5e-4,
            "Metal gated_rms_norm mean_abs_diff={mean:e} exceeds 5e-4"
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_rms_norm_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!("Metal unavailable, skipping test_metal_rms_norm_matches_fallback");
            return Ok(());
        };

        let batch = 2usize;
        let seq_len = 3usize;
        let hidden = 4096usize;
        let elems = batch * seq_len * hidden;

        let mut rng = StdRng::seed_from_u64(0xA11CE);
        let x_data: Vec<f32> = (0..elems)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let w_data: Vec<f32> = (0..hidden)
            .map(|_| rng.random_range(-0.2f32..0.2f32))
            .collect();

        let x = Tensor::from_slice(&x_data, (batch, seq_len, hidden), &device)?
            .to_dtype(DType::BF16)?;
        let weight = Tensor::from_slice(&w_data, (hidden,), &device)?.to_dtype(DType::BF16)?;

        assert!(crate::backend::metal::metal_rms_norm_supports(&x, &weight));
        let fallback = rms_norm_fallback(&x, &weight, 1e-6)?;
        let fused = crate::backend::metal::metal_rms_norm_bf16(&x, &weight, 1e-6)?;

        assert_eq!(fused.dims(), fallback.dims());
        assert_eq!(fused.dtype(), DType::BF16);

        let diff = (fused.to_dtype(DType::F32)?
            - fallback.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!("rms_norm metal vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
        assert!(
            max < 5e-3,
            "Metal rms_norm max_abs_diff={max:e} exceeds 5e-3"
        );
        assert!(
            mean < 5e-4,
            "Metal rms_norm mean_abs_diff={mean:e} exceeds 5e-4"
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_lm_head_forward_decode_batch_matches_broadcast_matmul() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!(
                "Metal unavailable, skipping test_metal_lm_head_forward_decode_batch_matches_broadcast_matmul"
            );
            return Ok(());
        };

        let batch = 4usize;
        let hidden = 128usize;
        let vocab = 257usize;
        let x_data: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.0234375)
            .collect();
        let weight_data: Vec<f32> = (0..hidden * vocab)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.01953125)
            .collect();

        let x = Tensor::from_slice(&x_data, (batch, 1usize, hidden), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let weight_t = Tensor::from_slice(&weight_data, (hidden, vocab), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;

        assert!(
            crate::backend::metal::metal_transposed_coop_gemv_decode_batch_supports(&x, &weight_t)
        );
        let reference = x.broadcast_matmul(&weight_t)?;
        let fast = lm_head_forward(&x, &weight_t)?;

        assert_eq!(fast.dims(), &[batch, 1usize, vocab]);
        let diff = (fast.to_dtype(DType::F32)?
            - reference.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        assert!(
            max < 2e-2,
            "Metal batch LM-head max_abs_diff={max:e} exceeds 2e-2"
        );
        assert!(
            mean < 2e-3,
            "Metal batch LM-head mean_abs_diff={mean:e} exceeds 2e-3"
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_rotary_embedding_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!("Metal unavailable, skipping test_metal_rotary_embedding_matches_fallback");
            return Ok(());
        };

        let batch = 1usize;
        let seq_len = 5usize;
        let q_heads = 4usize;
        let k_heads = 2usize;
        let head_dim = 16usize;
        let rotary_dim = 8usize;
        let mut rng = StdRng::seed_from_u64(0xA07A_7E55);
        let q_data: Vec<f32> = (0..batch * seq_len * q_heads * head_dim)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let k_data: Vec<f32> = (0..batch * seq_len * k_heads * head_dim)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let q = Tensor::from_slice(&q_data, (batch, seq_len, q_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::from_slice(&k_data, (batch, seq_len, k_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let positions: Vec<f32> = (11..11 + seq_len).map(|p| p as f32).collect();
        let positions = Tensor::from_slice(&positions, (seq_len,), &device)?;
        let inv_freq = compute_rotary_inv_freq(rotary_dim, 10_000.0, &device)?;
        let (cos, sin) = rotary_tables_from_tensor(&positions, &inv_freq)?;

        assert!(crate::backend::metal::metal_rotary_embedding_supports(
            &q, &k, &cos, &sin, head_dim, rotary_dim,
        ));
        let (q_fused, k_fused) = crate::backend::metal::metal_rotary_embedding_bf16(
            &q, &k, &cos, &sin, head_dim, rotary_dim,
        )?;
        let q_ref = apply_rope(&q, &cos, &sin, head_dim, rotary_dim)?;
        let k_ref = apply_rope(&k, &cos, &sin, head_dim, rotary_dim)?;

        let q_diff = (q_fused.to_dtype(DType::F32)?
            - q_ref.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?
        .abs()?;
        let k_diff = (k_fused.to_dtype(DType::F32)?
            - k_ref.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?
        .abs()?;
        let q_max = q_diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let k_max = k_diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(q_max < 1e-6, "Metal rotary Q max_abs_diff={q_max:e}");
        assert!(k_max < 1e-6, "Metal rotary K max_abs_diff={k_max:e}");

        Ok(())
    }

    #[test]
    fn test_rope_preserves_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 8;

        let q = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, num_heads, head_dim), &device)?;
        let k = Tensor::randn(
            0.0_f32,
            1.0,
            (batch, seq_len, num_kv_heads, head_dim),
            &device,
        )?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let (rq, rk) = rotary_embedding(&q, &k, &positions, head_dim, head_dim, &inv_freq)?;

        assert_eq!(rq.dims(), &[batch, seq_len, num_heads, head_dim]);
        assert_eq!(rk.dims(), &[batch, seq_len, num_kv_heads, head_dim]);

        Ok(())
    }

    #[test]
    fn test_rope_position_zero_is_identity() -> Result<()> {
        let device = Device::Cpu;
        // At position 0, cos=1 and sin=0, so rotation should be identity
        let head_dim = 4;
        let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let q = Tensor::new(q_data.as_slice(), &device)?.reshape((1, 1, 1, head_dim))?;
        let k = q.clone();

        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let (rq, _rk) = rotary_embedding(&q, &k, &[0], head_dim, head_dim, &inv_freq)?;
        let orig = q.flatten_all()?.to_vec1::<f32>()?;
        let rotated = rq.flatten_all()?.to_vec1::<f32>()?;

        for i in 0..head_dim {
            assert!(
                (orig[i] - rotated[i]).abs() < 1e-5,
                "Position 0 should be identity, dim {i}: orig={} rotated={}",
                orig[i],
                rotated[i]
            );
        }

        Ok(())
    }

    #[test]
    fn test_rope_different_positions_differ() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 8;
        let q = Tensor::ones((1, 2, 1, head_dim), DType::F32, &device)?;
        let k = q.clone();

        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let (rq, _) = rotary_embedding(&q, &k, &[0, 100], head_dim, head_dim, &inv_freq)?;
        // rq shape: [1, 2, 1, 8] — extract pos 0 and pos 100
        let pos0 = rq.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
        let pos100 = rq.narrow(1, 1, 1)?.flatten_all()?.to_vec1::<f32>()?;

        let diff: f32 = pos0.iter().zip(&pos100).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.01,
            "Different positions should produce different embeddings"
        );

        Ok(())
    }

    #[test]
    fn test_partial_rope_passthrough_dims_unchanged() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 8;
        let rotary_dim = 4; // only rotate first 4 dims, last 4 pass through
        let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let q = Tensor::new(q_data.as_slice(), &device)?.reshape((1, 1, 1, head_dim))?;
        let k = q.clone();

        // Position 100 — the rotary dims should change, passthrough dims should not
        let inv_freq = compute_rotary_inv_freq(rotary_dim, 10_000.0, &device)?;
        let (rq, _) = rotary_embedding(&q, &k, &[100], head_dim, rotary_dim, &inv_freq)?;
        let orig = q.flatten_all()?.to_vec1::<f32>()?;
        let rotated = rq.flatten_all()?.to_vec1::<f32>()?;

        // First rotary_dim dims should be different at non-zero position
        let rotary_diff: f32 = (0..rotary_dim).map(|i| (orig[i] - rotated[i]).abs()).sum();
        assert!(
            rotary_diff > 0.01,
            "Rotary dims should change at position 100"
        );

        // Passthrough dims (rotary_dim..head_dim) must be identical
        for i in rotary_dim..head_dim {
            assert!(
                (orig[i] - rotated[i]).abs() < 1e-6,
                "Passthrough dim {i} should be unchanged: orig={} rotated={}",
                orig[i],
                rotated[i]
            );
        }

        Ok(())
    }

    #[test]
    fn test_partial_rope_preserves_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 16;
        let rotary_dim = 4; // partial rotation

        let q = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, num_heads, head_dim), &device)?;
        let k = Tensor::randn(
            0.0_f32,
            1.0,
            (batch, seq_len, num_kv_heads, head_dim),
            &device,
        )?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let inv_freq = compute_rotary_inv_freq(rotary_dim, 10_000.0, &device)?;
        let (rq, rk) = rotary_embedding(&q, &k, &positions, head_dim, rotary_dim, &inv_freq)?;

        assert_eq!(rq.dims(), &[batch, seq_len, num_heads, head_dim]);
        assert_eq!(rk.dims(), &[batch, seq_len, num_kv_heads, head_dim]);

        Ok(())
    }

    #[test]
    fn test_swiglu_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let seq_len = 3;
        let hidden = 4;
        let intermediate = 8;

        let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), &device)?;
        let gate = Tensor::randn(0.0_f32, 0.1, (intermediate, hidden), &device)?;
        let up = Tensor::randn(0.0_f32, 0.1, (intermediate, hidden), &device)?;
        let down = Tensor::randn(0.0_f32, 0.1, (hidden, intermediate), &device)?;
        let gate_t = gate.t()?.contiguous()?;
        let up_t = up.t()?.contiguous()?;
        let down_t = down.t()?.contiguous()?;

        let mlp = GpuFfnWeights {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
            gate_proj_t: gate_t,
            up_proj_t: up_t,
            down_proj_t: down_t,
            gate_proj_marlin: None,
            up_proj_marlin: None,
            down_proj_marlin: None,
        };
        let result = swiglu_ffn(&x, &mlp, None)?;
        assert_eq!(result.dims(), &[batch, seq_len, hidden]);

        Ok(())
    }

    #[test]
    fn test_swiglu_zero_gate_gives_zero() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 4;
        let intermediate = 8;

        let x = Tensor::ones((1, 1, hidden), DType::F32, &device)?;
        // Gate weights all zero -> silu(0) = 0 -> output is zero regardless of up/down
        let gate = Tensor::zeros((intermediate, hidden), DType::F32, &device)?;
        let up = Tensor::ones((intermediate, hidden), DType::F32, &device)?;
        let down = Tensor::ones((hidden, intermediate), DType::F32, &device)?;
        let gate_t = gate.t()?.contiguous()?;
        let up_t = up.t()?.contiguous()?;
        let down_t = down.t()?.contiguous()?;

        let mlp = GpuFfnWeights {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
            gate_proj_t: gate_t,
            up_proj_t: up_t,
            down_proj_t: down_t,
            gate_proj_marlin: None,
            up_proj_marlin: None,
            down_proj_marlin: None,
        };
        let result = swiglu_ffn(&x, &mlp, None)?;
        let vals = result.to_vec3::<f32>()?;

        for v in &vals[0][0] {
            assert!(
                v.abs() < 1e-6,
                "SwiGLU with zero gate should produce zero, got {v}"
            );
        }

        Ok(())
    }

    /// Create a minimal config for tests (no output gate, simple dims).
    fn make_test_config(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden: usize,
    ) -> kiln_core::config::ModelConfig {
        kiln_core::config::ModelConfig {
            hidden_size: hidden,
            num_layers: 4,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size: hidden * 2,
            vocab_size: 256,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            dtype: kiln_core::config::DType::BF16,
            num_full_attention_layers: 1,
            full_attention_interval: 4,
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0, // tests use full rotation by default
        }
    }

    #[test]
    fn test_linear_attention_state_prefix_snapshot_truncates_draft_state() -> Result<()> {
        let device = Device::Cpu;
        let config = make_test_config(2, 1, 4, 8);
        let state = LinearAttentionState::new(&config, &device)?;

        assert_eq!(state.recurrent_states.len(), 3);
        assert_eq!(state.conv_states.len(), 3);

        let draft = state.snapshot_for_decode_rollback_prefix(1)?;
        assert_eq!(draft.recurrent_states.len(), 1);
        assert_eq!(draft.conv_states.len(), 1);
        Ok(())
    }

    fn make_test_attn_weights(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden: usize,
        device: &Device,
    ) -> Result<GpuFullAttentionWeights> {
        let q_proj = Tensor::randn(0.0_f32, 0.02, (num_heads * head_dim, hidden), device)?;
        let k_proj = Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, hidden), device)?;
        let v_proj = Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, hidden), device)?;
        let o_proj = Tensor::randn(0.0_f32, 0.02, (hidden, num_heads * head_dim), device)?;
        let q_proj_t = q_proj.t()?.contiguous()?;
        let k_proj_t = k_proj.t()?.contiguous()?;
        let v_proj_t = v_proj.t()?.contiguous()?;
        let o_proj_t = o_proj.t()?.contiguous()?;
        Ok(GpuFullAttentionWeights {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
            k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
            q_proj_t,
            k_proj_t,
            v_proj_t,
            qkv_proj_t: None,
            o_proj_t,
            q_proj_marlin: None,
        })
    }

    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn patterned_bf16(shape: &[usize], scale: f32, device: &Device) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| (((i * 17 + 13) % 257) as f32 - 128.0) * scale)
            .collect();
        Ok(Tensor::new(data, device)?
            .reshape(shape)?
            .to_dtype(DType::BF16)?
            .contiguous()?)
    }

    #[cfg(feature = "metal")]
    fn patterned_f32(shape: &[usize], scale: f32, device: &Device) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| (((i * 23 + 19) % 251) as f32 - 125.0) * scale)
            .collect();
        Ok(Tensor::new(data, device)?.reshape(shape)?.contiguous()?)
    }

    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn make_bf16_full_attn_weights(
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<GpuFullAttentionWeights> {
        let q_proj = patterned_bf16(&[num_heads * head_dim, hidden], 0.00002, device)?;
        let k_proj = patterned_bf16(&[num_kv_heads * head_dim, hidden], 0.00003, device)?;
        let v_proj = patterned_bf16(&[num_kv_heads * head_dim, hidden], 0.00004, device)?;
        let o_proj = patterned_bf16(&[hidden, num_heads * head_dim], 0.00002, device)?;
        Ok(GpuFullAttentionWeights {
            q_proj_t: q_proj.t()?.contiguous()?,
            k_proj_t: k_proj.t()?.contiguous()?,
            v_proj_t: v_proj.t()?.contiguous()?,
            qkv_proj_t: None,
            o_proj_t: o_proj.t()?.contiguous()?,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
            k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
            q_proj_marlin: None,
        })
    }

    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn make_bf16_mlp_weights(
        hidden: usize,
        intermediate: usize,
        device: &Device,
    ) -> Result<GpuFfnWeights> {
        let gate_proj = patterned_bf16(&[intermediate, hidden], 0.00003, device)?;
        let up_proj = patterned_bf16(&[intermediate, hidden], 0.00002, device)?;
        let down_proj = patterned_bf16(&[hidden, intermediate], 0.00003, device)?;
        Ok(GpuFfnWeights {
            gate_proj_t: gate_proj.t()?.contiguous()?,
            up_proj_t: up_proj.t()?.contiguous()?,
            down_proj_t: down_proj.t()?.contiguous()?,
            gate_proj,
            up_proj,
            down_proj,
            gate_proj_marlin: None,
            up_proj_marlin: None,
            down_proj_marlin: None,
        })
    }

    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn make_bf16_full_attention_gpu_weights(
        vocab: usize,
        hidden: usize,
        intermediate: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_layers: usize,
        device: &Device,
    ) -> Result<GpuWeights> {
        let embed_tokens = patterned_bf16(&[vocab, hidden], 0.01, device)?;
        let embed_tokens_t = embed_tokens.t()?.contiguous()?;
        let final_norm = Tensor::zeros(hidden, DType::F32, device)?;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(GpuLayerWeights {
                input_layernorm: Tensor::zeros(hidden, DType::F32, device)?,
                post_attention_layernorm: Tensor::zeros(hidden, DType::F32, device)?,
                attention: GpuAttentionWeights::Full(make_bf16_full_attn_weights(
                    hidden,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    device,
                )?),
                mlp: make_bf16_mlp_weights(hidden, intermediate, device)?,
            });
        }
        let rotary_inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, device)?;
        Ok(GpuWeights {
            embed_tokens,
            embed_tokens_t,
            layers,
            final_norm,
            rotary_inv_freq,
            mtp: None,
        })
    }

    #[cfg(feature = "metal")]
    fn make_bf16_hybrid_gpu_weights(
        config: &kiln_core::config::ModelConfig,
        device: &Device,
    ) -> Result<GpuWeights> {
        let hidden = config.hidden_size;
        let embed_tokens = patterned_bf16(&[config.vocab_size, hidden], 0.01, device)?;
        let embed_tokens_t = embed_tokens.t()?.contiguous()?;
        let final_norm = Tensor::zeros(hidden, DType::BF16, device)?;
        let mut layers = Vec::with_capacity(config.num_layers);

        for layer_idx in 0..config.num_layers {
            let attention = if config.is_full_attention_layer(layer_idx) {
                let q_dim = config.full_attn_q_proj_dim();
                let kv_dim = config.num_kv_heads * config.head_dim;
                let out_dim = config.num_attention_heads * config.head_dim;
                let q_proj = patterned_bf16(&[q_dim, hidden], 0.00002, device)?;
                let k_proj = patterned_bf16(&[kv_dim, hidden], 0.00003, device)?;
                let v_proj = patterned_bf16(&[kv_dim, hidden], 0.00004, device)?;
                let o_proj = patterned_bf16(&[hidden, out_dim], 0.00002, device)?;
                GpuAttentionWeights::Full(GpuFullAttentionWeights {
                    q_proj_t: q_proj.t()?.contiguous()?,
                    k_proj_t: k_proj.t()?.contiguous()?,
                    v_proj_t: v_proj.t()?.contiguous()?,
                    qkv_proj_t: None,
                    o_proj_t: o_proj.t()?.contiguous()?,
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm: Tensor::zeros(config.head_dim, DType::BF16, device)?,
                    k_norm: Tensor::zeros(config.head_dim, DType::BF16, device)?,
                    q_proj_marlin: None,
                })
            } else {
                let qkv_dim = config.linear_qkv_dim();
                let v_dim = config.linear_v_dim();
                let nv = config.linear_num_value_heads;
                let in_proj_qkv = patterned_bf16(&[qkv_dim, hidden], 0.00002, device)?;
                let in_proj_z = patterned_bf16(&[v_dim, hidden], 0.00003, device)?;
                let out_proj = patterned_bf16(&[hidden, v_dim], 0.00002, device)?;
                let in_proj_a = patterned_bf16(&[nv, hidden], 0.00004, device)?;
                let in_proj_b = patterned_bf16(&[nv, hidden], 0.00005, device)?;
                GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                    in_proj_qkv_t: in_proj_qkv.t()?.contiguous()?,
                    in_proj_z_t: in_proj_z.t()?.contiguous()?,
                    in_proj_a_t: in_proj_a.t()?.contiguous()?,
                    in_proj_b_t: in_proj_b.t()?.contiguous()?,
                    in_proj_ab_t: None,
                    out_proj_t: out_proj.t()?.contiguous()?,
                    in_proj_qkv,
                    in_proj_z,
                    out_proj,
                    in_proj_a,
                    in_proj_b,
                    conv1d: patterned_bf16(
                        &[qkv_dim, 1usize, config.linear_conv_kernel_dim],
                        0.00002,
                        device,
                    )?,
                    norm: Tensor::ones(config.linear_value_head_dim, DType::F32, device)?,
                    a_log: Tensor::zeros(nv, DType::F32, device)?,
                    a_log_gates: Tensor::zeros(nv, DType::F32, device)?,
                    dt_bias: Tensor::zeros(nv, DType::BF16, device)?,
                    out_proj_marlin: None,
                })
            };

            layers.push(GpuLayerWeights {
                input_layernorm: Tensor::zeros(hidden, DType::BF16, device)?,
                post_attention_layernorm: Tensor::zeros(hidden, DType::BF16, device)?,
                attention,
                mlp: make_bf16_mlp_weights(hidden, config.intermediate_size, device)?,
            });
        }

        let rotary_inv_freq =
            compute_rotary_inv_freq(config.rotary_dim(), config.rope_theta, device)?;
        Ok(GpuWeights {
            embed_tokens,
            embed_tokens_t,
            layers,
            final_norm,
            rotary_inv_freq,
            mtp: None,
        })
    }

    #[cfg(feature = "metal")]
    fn patterned_linear_state(
        config: &kiln_core::config::ModelConfig,
        row: usize,
        device: &Device,
    ) -> Result<LinearAttentionState> {
        let mut state = LinearAttentionState::new(config, device)?;
        for layer_idx in 0..state.recurrent_states.len() {
            let state_scale = 0.00001 * (row + layer_idx + 1) as f32;
            let conv_scale = 0.00002 * (row + layer_idx + 1) as f32;
            state.recurrent_states[layer_idx] = patterned_bf16(
                &[
                    1usize,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                ],
                state_scale,
                device,
            )?;
            state.conv_states[layer_idx] = patterned_f32(
                &[
                    1usize,
                    config.linear_qkv_dim(),
                    config.linear_conv_kernel_dim - 1,
                ],
                conv_scale,
                device,
            )?;
        }
        Ok(state)
    }

    #[cfg(feature = "metal")]
    fn tensor_abs_diff_stats(left: &Tensor, right: &Tensor) -> Result<(f32, f32)> {
        let diff = (left.to_dtype(DType::F32)? - right.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        Ok((max, mean))
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_gqa_attention_paged_decode_contiguous_batch_matches_rowwise_metal() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };
        if std::env::var("KILN_DISABLE_FUSED_PAGED_DECODE").is_ok() {
            eprintln!("fused paged decode disabled; skipping batched contiguous decode test");
            return Ok(());
        }

        let backend = crate::backend::for_device(&device);
        let batch = 2usize;
        let hidden = 512usize;
        let num_heads = 16usize;
        let num_kv_heads = 4usize;
        let head_dim = 256usize;
        let block_size = 16usize;
        let start_pos = 3usize;

        let attn = make_bf16_full_attn_weights(hidden, num_heads, num_kv_heads, head_dim, &device)?;

        let x = patterned_bf16(&[batch, 1usize, hidden], 0.01, &device)?;
        let positions = Tensor::from_slice(&[start_pos as f32], 1usize, &device)?;
        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;

        let prefix_k = patterned_bf16(&[batch, start_pos, num_kv_heads, head_dim], 0.002, &device)?;
        let prefix_v = patterned_bf16(&[batch, start_pos, num_kv_heads, head_dim], 0.003, &device)?;
        let bt0 = BlockTable { blocks: vec![0] };
        let bt1 = BlockTable { blocks: vec![1] };
        let block_tables = [&bt0, &bt1];
        let start_positions = [start_pos, start_pos];
        let mut batch_cache = PagedKvCache::new(
            1,
            2,
            block_size,
            num_kv_heads,
            head_dim,
            DType::BF16,
            &device,
        )?;
        for (row, block_table) in block_tables.iter().enumerate() {
            let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
            let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
            assert!(batch_cache.write_token_major_native(0, block_table, 0, &row_k, &row_v)?);
        }

        let batched = gqa_attention_paged_decode_contiguous_batch(
            &*backend,
            &x,
            &attn,
            &positions,
            &start_positions,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            &inv_freq,
            1e-6,
            &mut batch_cache,
            &block_tables,
            0,
            false,
            None,
            None,
            None,
        )?;
        device.synchronize()?;
        assert_eq!(batched.dims(), &[batch, 1usize, hidden]);

        for row in 0..batch {
            let mut row_cache = PagedKvCache::new(
                1,
                1,
                block_size,
                num_kv_heads,
                head_dim,
                DType::BF16,
                &device,
            )?;
            let row_table = BlockTable { blocks: vec![0] };
            let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
            let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
            assert!(row_cache.write_token_major_native(0, &row_table, 0, &row_k, &row_v)?);
            let row_x = x.narrow(0, row, 1)?.contiguous()?;
            let rowwise = gqa_attention_paged(
                &*backend,
                &row_x,
                &attn,
                &positions,
                start_pos,
                num_heads,
                num_kv_heads,
                head_dim,
                head_dim,
                &inv_freq,
                1e-6,
                &mut row_cache,
                &row_table,
                0,
                false,
                None,
            )?;
            device.synchronize()?;

            let batch_row = batched.narrow(0, row, 1)?;
            let diff = (batch_row.to_dtype(DType::F32)? - rowwise.to_dtype(DType::F32)?)?;
            let abs = diff.abs()?;
            let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
            eprintln!(
                "batched contiguous paged decode row {row}: max_abs_diff={max:e} mean_abs_diff={mean:e}"
            );
            assert!(
                max <= 2e-2,
                "row {row} batched contiguous paged decode max_abs_diff={max:e}"
            );
            assert!(
                mean <= 2e-3,
                "row {row} batched contiguous paged decode mean_abs_diff={mean:e}"
            );
        }

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_transformer_block_paged_decode_contiguous_batch_matches_rowwise_metal() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };
        if std::env::var("KILN_DISABLE_FUSED_PAGED_DECODE").is_ok() {
            eprintln!("fused paged decode disabled; skipping batched contiguous transformer test");
            return Ok(());
        }

        let backend = crate::backend::for_device(&device);
        let batch = 2usize;
        let hidden = 512usize;
        let intermediate = 768usize;
        let num_heads = 16usize;
        let num_kv_heads = 4usize;
        let head_dim = 256usize;
        let block_size = 16usize;
        let start_pos = 3usize;
        let config = kiln_core::config::ModelConfig {
            hidden_size: hidden,
            num_layers: 1,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size: intermediate,
            vocab_size: 1024,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            dtype: kiln_core::config::DType::BF16,
            num_full_attention_layers: 1,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: num_kv_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let layer = GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            post_attention_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            attention: GpuAttentionWeights::Full(make_bf16_full_attn_weights(
                hidden,
                num_heads,
                num_kv_heads,
                head_dim,
                &device,
            )?),
            mlp: make_bf16_mlp_weights(hidden, intermediate, &device)?,
        };
        let x = patterned_bf16(&[batch, 1usize, hidden], 0.01, &device)?;
        let positions = Tensor::from_slice(&[start_pos as f32], 1usize, &device)?;
        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let prefix_k = patterned_bf16(&[batch, start_pos, num_kv_heads, head_dim], 0.002, &device)?;
        let prefix_v = patterned_bf16(&[batch, start_pos, num_kv_heads, head_dim], 0.003, &device)?;
        let bt0 = BlockTable { blocks: vec![0] };
        let bt1 = BlockTable { blocks: vec![1] };
        let block_tables = [&bt0, &bt1];
        let start_positions = [start_pos, start_pos];
        let mut batch_cache = PagedKvCache::new(
            1,
            2,
            block_size,
            num_kv_heads,
            head_dim,
            DType::BF16,
            &device,
        )?;
        for (row, block_table) in block_tables.iter().enumerate() {
            let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
            let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
            assert!(batch_cache.write_token_major_native(0, block_table, 0, &row_k, &row_v)?);
        }

        let batched = transformer_block_paged_decode_contiguous_batch(
            &*backend,
            &x,
            &layer,
            &config,
            &positions,
            &start_positions,
            &inv_freq,
            &mut batch_cache,
            &block_tables,
            0,
            None,
            None,
            None,
        )?;
        device.synchronize()?;
        assert_eq!(batched.dims(), &[batch, 1usize, hidden]);

        for row in 0..batch {
            let mut row_cache = PagedKvCache::new(
                1,
                1,
                block_size,
                num_kv_heads,
                head_dim,
                DType::BF16,
                &device,
            )?;
            let row_table = BlockTable { blocks: vec![0] };
            let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
            let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
            assert!(row_cache.write_token_major_native(0, &row_table, 0, &row_k, &row_v)?);
            let row_x = x.narrow(0, row, 1)?.contiguous()?;
            let rowwise = transformer_block_paged(
                &*backend,
                &row_x,
                &layer,
                &config,
                &positions,
                start_pos,
                num_heads,
                num_kv_heads,
                head_dim,
                head_dim,
                &inv_freq,
                config.rms_norm_eps,
                &mut row_cache,
                &row_table,
                0,
                None,
            )?;
            device.synchronize()?;

            let batch_row = batched.narrow(0, row, 1)?;
            let diff = (batch_row.to_dtype(DType::F32)? - rowwise.to_dtype(DType::F32)?)?;
            let abs = diff.abs()?;
            let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
            eprintln!(
                "batched contiguous transformer block row {row}: max_abs_diff={max:e} mean_abs_diff={mean:e}"
            );
            assert!(
                max <= 3e-2,
                "row {row} batched contiguous transformer block max_abs_diff={max:e}"
            );
            assert!(
                mean <= 3e-3,
                "row {row} batched contiguous transformer block mean_abs_diff={mean:e}"
            );
        }

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_model_forward_paged_decode_contiguous_batch_matches_rowwise_metal() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };
        if std::env::var("KILN_DISABLE_FUSED_PAGED_DECODE").is_ok() {
            eprintln!("fused paged decode disabled; skipping batched contiguous model test");
            return Ok(());
        }

        let backend = crate::backend::for_device(&device);
        let batch = 2usize;
        let vocab = 64usize;
        let hidden = 512usize;
        let intermediate = 768usize;
        let num_heads = 16usize;
        let num_kv_heads = 4usize;
        let head_dim = 256usize;
        let block_size = 16usize;
        let start_pos = 3usize;
        let weights = make_bf16_full_attention_gpu_weights(
            vocab,
            hidden,
            intermediate,
            num_heads,
            num_kv_heads,
            head_dim,
            1,
            &device,
        )?;
        let config = kiln_core::config::ModelConfig {
            hidden_size: hidden,
            num_layers: 1,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size: intermediate,
            vocab_size: vocab,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            dtype: kiln_core::config::DType::BF16,
            num_full_attention_layers: 1,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: num_kv_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let prefix_k = patterned_bf16(&[batch, start_pos, num_kv_heads, head_dim], 0.002, &device)?;
        let prefix_v = patterned_bf16(&[batch, start_pos, num_kv_heads, head_dim], 0.003, &device)?;
        let bt0 = BlockTable { blocks: vec![0] };
        let bt1 = BlockTable { blocks: vec![1] };
        let block_tables = [&bt0, &bt1];
        let start_positions = [start_pos, start_pos];
        let token_ids = [7u32, 11u32];
        let mut batch_cache = PagedKvCache::new(
            1,
            2,
            block_size,
            num_kv_heads,
            head_dim,
            DType::BF16,
            &device,
        )?;
        for (row, block_table) in block_tables.iter().enumerate() {
            let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
            let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
            assert!(batch_cache.write_token_major_native(0, block_table, 0, &row_k, &row_v)?);
        }

        let batched = model_forward_paged_decode_contiguous_batch(
            &*backend,
            &token_ids,
            &weights,
            &config,
            &mut batch_cache,
            &block_tables,
            &start_positions,
            None,
            None,
        )?;
        device.synchronize()?;
        assert_eq!(batched.dims(), &[batch, 1usize, vocab]);

        let positions = Tensor::from_slice(&[start_pos as f32], 1usize, &device)?;
        for row in 0..batch {
            let mut row_cache = PagedKvCache::new(
                1,
                1,
                block_size,
                num_kv_heads,
                head_dim,
                DType::BF16,
                &device,
            )?;
            let row_table = BlockTable { blocks: vec![0] };
            let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
            let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
            assert!(row_cache.write_token_major_native(0, &row_table, 0, &row_k, &row_v)?);
            let rowwise = model_forward_paged(
                &*backend,
                &token_ids[row..row + 1],
                &weights,
                &config,
                &mut row_cache,
                &row_table,
                start_pos,
                None,
                None,
                Some(&positions),
            )?;
            device.synchronize()?;

            let batch_row = batched.narrow(0, row, 1)?;
            let diff = (batch_row.to_dtype(DType::F32)? - rowwise.to_dtype(DType::F32)?)?;
            let abs = diff.abs()?;
            let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
            eprintln!(
                "batched contiguous model decode row {row}: max_abs_diff={max:e} mean_abs_diff={mean:e}"
            );
            assert!(
                max <= 3e-2,
                "row {row} batched contiguous model decode max_abs_diff={max:e}"
            );
            assert!(
                mean <= 3e-3,
                "row {row} batched contiguous model decode mean_abs_diff={mean:e}"
            );
        }

        Ok(())
    }

    /// Phase 12-B-prime: parity test that exercises the dyn_seqlen varlen
    /// paged decode path under a non-uniform `start_positions` batch on CUDA.
    /// The Metal-gated test above only covers the uniform `start_pos`
    /// fast-path; this test confirms that batched decode with divergent
    /// per-row K/V prefix lengths still matches per-row `model_forward_paged`
    /// bit-for-bit (within bf16 numeric tolerance).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_model_forward_paged_decode_contiguous_batch_dyn_seqlen_cuda() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!(
                    "CUDA unavailable, skipping test_model_forward_paged_decode_contiguous_batch_dyn_seqlen_cuda: {err}"
                );
                return Ok(());
            }
        };
        if std::env::var("KILN_DISABLE_FUSED_PAGED_DECODE").is_ok() {
            eprintln!("fused paged decode disabled; skipping dyn_seqlen batched test");
            return Ok(());
        }

        let backend = crate::backend::for_device(&device);
        let batch = 2usize;
        let vocab = 64usize;
        let hidden = 512usize;
        let intermediate = 768usize;
        let num_heads = 16usize;
        let num_kv_heads = 4usize;
        let head_dim = 256usize;
        let block_size = 16usize;
        // Non-uniform start positions — the whole point of this test.
        let start_positions = [3usize, 5usize];
        let weights = make_bf16_full_attention_gpu_weights(
            vocab,
            hidden,
            intermediate,
            num_heads,
            num_kv_heads,
            head_dim,
            1,
            &device,
        )?;
        let config = kiln_core::config::ModelConfig {
            hidden_size: hidden,
            num_layers: 1,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size: intermediate,
            vocab_size: vocab,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            dtype: kiln_core::config::DType::BF16,
            num_full_attention_layers: 1,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: num_kv_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        // Build per-row prefix K/V at each row's actual start_pos so that
        // the batched cache holds divergent K/V prefix lengths. Distinct
        // patterns per row catch any cross-row leakage.
        let prefix_k_row0 = patterned_bf16(
            &[1, start_positions[0], num_kv_heads, head_dim],
            0.002,
            &device,
        )?;
        let prefix_v_row0 = patterned_bf16(
            &[1, start_positions[0], num_kv_heads, head_dim],
            0.003,
            &device,
        )?;
        let prefix_k_row1 = patterned_bf16(
            &[1, start_positions[1], num_kv_heads, head_dim],
            0.0021,
            &device,
        )?;
        let prefix_v_row1 = patterned_bf16(
            &[1, start_positions[1], num_kv_heads, head_dim],
            0.0031,
            &device,
        )?;

        let bt0 = BlockTable { blocks: vec![0] };
        let bt1 = BlockTable { blocks: vec![1] };
        let block_tables = [&bt0, &bt1];
        let token_ids = [7u32, 11u32];

        let mut batch_cache = PagedKvCache::new(
            1,
            2,
            block_size,
            num_kv_heads,
            head_dim,
            DType::BF16,
            &device,
        )?;
        assert!(batch_cache.write_token_major_native(
            0,
            &bt0,
            0,
            &prefix_k_row0,
            &prefix_v_row0
        )?);
        assert!(batch_cache.write_token_major_native(
            0,
            &bt1,
            0,
            &prefix_k_row1,
            &prefix_v_row1
        )?);

        let batched = model_forward_paged_decode_contiguous_batch(
            &*backend,
            &token_ids,
            &weights,
            &config,
            &mut batch_cache,
            &block_tables,
            &start_positions,
            None,
            None,
        )?;
        device.synchronize()?;
        assert_eq!(batched.dims(), &[batch, 1usize, vocab]);

        for row in 0..batch {
            let row_start_pos = start_positions[row];
            let mut row_cache = PagedKvCache::new(
                1,
                1,
                block_size,
                num_kv_heads,
                head_dim,
                DType::BF16,
                &device,
            )?;
            let row_table = BlockTable { blocks: vec![0] };
            let (row_k, row_v) = if row == 0 {
                (prefix_k_row0.clone(), prefix_v_row0.clone())
            } else {
                (prefix_k_row1.clone(), prefix_v_row1.clone())
            };
            assert!(row_cache.write_token_major_native(0, &row_table, 0, &row_k, &row_v)?);
            let positions = Tensor::from_slice(&[row_start_pos as f32], 1usize, &device)?;
            let rowwise = model_forward_paged(
                &*backend,
                &token_ids[row..row + 1],
                &weights,
                &config,
                &mut row_cache,
                &row_table,
                row_start_pos,
                None,
                None,
                Some(&positions),
            )?;
            device.synchronize()?;

            let batch_row = batched.narrow(0, row, 1)?;
            let diff = (batch_row.to_dtype(DType::F32)? - rowwise.to_dtype(DType::F32)?)?;
            let abs = diff.abs()?;
            let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
            eprintln!(
                "dyn_seqlen batched contiguous model decode row {row} (start_pos={row_start_pos}): max_abs_diff={max:e} mean_abs_diff={mean:e}"
            );
            assert!(
                max <= 3e-2,
                "row {row} dyn_seqlen batched contiguous model decode max_abs_diff={max:e}"
            );
            assert!(
                mean <= 3e-3,
                "row {row} dyn_seqlen batched contiguous model decode mean_abs_diff={mean:e}"
            );
        }

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_model_forward_paged_decode_contiguous_batch_hybrid_matches_rowwise_metal() -> Result<()>
    {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };
        if std::env::var("KILN_DISABLE_FUSED_PAGED_DECODE").is_ok() {
            eprintln!("fused paged decode disabled; skipping batched hybrid model test");
            return Ok(());
        }

        let backend = crate::backend::for_device(&device);
        let batch = 2usize;
        let block_size = 16usize;
        let start_pos = 3usize;
        let config = kiln_core::config::ModelConfig {
            hidden_size: 256,
            num_layers: 4,
            num_attention_heads: 16,
            num_kv_heads: 4,
            head_dim: 256,
            intermediate_size: 512,
            vocab_size: 64,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            dtype: kiln_core::config::DType::BF16,
            num_full_attention_layers: 1,
            full_attention_interval: 4,
            attn_output_gate: true,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 0.25,
        };
        let weights = make_bf16_hybrid_gpu_weights(&config, &device)?;

        let prefix_k = patterned_bf16(
            &[batch, start_pos, config.num_kv_heads, config.head_dim],
            0.002,
            &device,
        )?;
        let prefix_v = patterned_bf16(
            &[batch, start_pos, config.num_kv_heads, config.head_dim],
            0.003,
            &device,
        )?;
        let bt0 = BlockTable { blocks: vec![0] };
        let bt1 = BlockTable { blocks: vec![1] };
        let block_tables = [&bt0, &bt1];
        let start_positions = [start_pos, start_pos];
        let token_ids = [7u32, 11u32];
        let mut batch_cache = PagedKvCache::new(
            config.num_full_attention_layers,
            2,
            block_size,
            config.num_kv_heads,
            config.head_dim,
            DType::BF16,
            &device,
        )?;
        for (row, block_table) in block_tables.iter().enumerate() {
            let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
            let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
            assert!(batch_cache.write_token_major_native(0, block_table, 0, &row_k, &row_v)?);
        }

        let mut row_states = Vec::with_capacity(batch);
        for row in 0..batch {
            row_states.push(patterned_linear_state(&config, row, &device)?);
        }
        let state_refs: Vec<&LinearAttentionState> = row_states.iter().collect();
        let mut batch_state = LinearAttentionState::from_batch_rows(&state_refs)?;
        let batched = model_forward_paged_decode_contiguous_batch(
            &*backend,
            &token_ids,
            &weights,
            &config,
            &mut batch_cache,
            &block_tables,
            &start_positions,
            Some(&mut batch_state),
            None,
        )?;
        device.synchronize()?;
        assert_eq!(batched.dims(), &[batch, 1usize, config.vocab_size]);

        let positions = Tensor::from_slice(&[start_pos as f32], 1usize, &device)?;
        for row in 0..batch {
            let mut row_cache = PagedKvCache::new(
                config.num_full_attention_layers,
                1,
                block_size,
                config.num_kv_heads,
                config.head_dim,
                DType::BF16,
                &device,
            )?;
            let row_table = BlockTable { blocks: vec![0] };
            let row_k = prefix_k.narrow(0, row, 1)?.contiguous()?;
            let row_v = prefix_v.narrow(0, row, 1)?.contiguous()?;
            assert!(row_cache.write_token_major_native(0, &row_table, 0, &row_k, &row_v)?);
            let rowwise = model_forward_paged(
                &*backend,
                &token_ids[row..row + 1],
                &weights,
                &config,
                &mut row_cache,
                &row_table,
                start_pos,
                Some(&mut row_states[row]),
                None,
                Some(&positions),
            )?;
            device.synchronize()?;

            let batch_row = batched.narrow(0, row, 1)?;
            let (max, mean) = tensor_abs_diff_stats(&batch_row, &rowwise)?;
            eprintln!(
                "batched hybrid model decode row {row}: max_abs_diff={max:e} mean_abs_diff={mean:e}"
            );
            assert!(
                max <= 5e-2,
                "row {row} batched hybrid model decode max_abs_diff={max:e}"
            );
            assert!(
                mean <= 5e-3,
                "row {row} batched hybrid model decode mean_abs_diff={mean:e}"
            );
        }

        let batch_rows = batch_state.split_batch_rows()?;
        for row in 0..batch {
            for layer_idx in 0..row_states[row].recurrent_states.len() {
                let (rec_max, rec_mean) = tensor_abs_diff_stats(
                    &batch_rows[row].recurrent_states[layer_idx],
                    &row_states[row].recurrent_states[layer_idx],
                )?;
                let (conv_max, conv_mean) = tensor_abs_diff_stats(
                    &batch_rows[row].conv_states[layer_idx],
                    &row_states[row].conv_states[layer_idx],
                )?;
                eprintln!(
                    "batched hybrid model state row {row} linear_layer {layer_idx}: recurrent_max={rec_max:e} recurrent_mean={rec_mean:e} conv_max={conv_max:e} conv_mean={conv_mean:e}"
                );
                assert!(
                    rec_max <= 5e-2,
                    "row {row} layer {layer_idx} recurrent max_abs_diff={rec_max:e}"
                );
                assert!(
                    rec_mean <= 5e-3,
                    "row {row} layer {layer_idx} recurrent mean_abs_diff={rec_mean:e}"
                );
                assert!(
                    conv_max <= 5e-2,
                    "row {row} layer {layer_idx} conv max_abs_diff={conv_max:e}"
                );
                assert!(
                    conv_mean <= 5e-3,
                    "row {row} layer {layer_idx} conv mean_abs_diff={conv_mean:e}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_gqa_attention_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq_len = 4;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let hidden = num_heads * head_dim; // 32

        let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), &device)?;
        let attn = make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let backend = test_backend(&device);
        let out = gqa_attention(
            &backend,
            &x,
            &attn,
            &positions,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            &inv_freq,
            1e-6,
            None,
            0,
            false,
            None,
        )?;
        assert_eq!(out.dims(), &[batch, seq_len, hidden]);

        Ok(())
    }

    #[test]
    fn test_gqa_head_expansion() -> Result<()> {
        // Verify GQA works: 4 Q heads, 2 KV heads (ratio=2)
        let device = Device::Cpu;
        let batch = 2;
        let seq_len = 3;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let hidden = num_heads * head_dim;

        let x = Tensor::randn(0.0_f32, 0.5, (batch, seq_len, hidden), &device)?;
        let attn = make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let backend = test_backend(&device);
        let out = gqa_attention(
            &backend,
            &x,
            &attn,
            &positions,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            &inv_freq,
            1e-6,
            None,
            0,
            false,
            None,
        )?;
        assert_eq!(out.dims(), &[batch, seq_len, hidden]);

        // Output should be finite and not all zeros
        let vals = out.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "output should be finite"
        );
        let sum: f32 = vals.iter().map(|v| v.abs()).sum();
        assert!(sum > 1e-6, "output should not be all zeros");

        Ok(())
    }

    #[test]
    fn test_gqa_single_token() -> Result<()> {
        // Single token should work (no causal masking needed)
        let device = Device::Cpu;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let hidden = num_heads * head_dim;

        let x = Tensor::randn(0.0_f32, 1.0, (1, 1, hidden), &device)?;
        let attn = make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?;

        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let backend = test_backend(&device);
        let out = gqa_attention(
            &backend,
            &x,
            &attn,
            &[0],
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            &inv_freq,
            1e-6,
            None,
            0,
            false,
            None,
        )?;
        assert_eq!(out.dims(), &[1, 1, hidden]);

        Ok(())
    }

    #[test]
    fn test_causal_mask() -> Result<()> {
        let device = Device::Cpu;
        // A 3x3 score matrix
        let scores = Tensor::ones((1, 1, 3, 3), DType::F32, &device)?;
        let masked = apply_causal_mask(&scores, 3)?;
        let vals = masked.flatten_all()?.to_vec1::<f32>()?;
        // Row 0: [1, -inf, -inf]
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!(vals[1].is_infinite() && vals[1] < 0.0);
        assert!(vals[2].is_infinite() && vals[2] < 0.0);
        // Row 1: [1, 1, -inf]
        assert!((vals[3] - 1.0).abs() < 1e-6);
        assert!((vals[4] - 1.0).abs() < 1e-6);
        assert!(vals[5].is_infinite() && vals[5] < 0.0);
        // Row 2: [1, 1, 1]
        assert!((vals[6] - 1.0).abs() < 1e-6);
        assert!((vals[7] - 1.0).abs() < 1e-6);
        assert!((vals[8] - 1.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_transformer_block_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq_len = 4;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let hidden = num_heads * head_dim;
        let intermediate = hidden * 2;

        let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), &device)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let gate_proj = Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?;
        let up_proj = Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?;
        let down_proj = Tensor::randn(0.0_f32, 0.02, (hidden, intermediate), &device)?;
        let gate_proj_t = gate_proj.t()?.contiguous()?;
        let up_proj_t = up_proj.t()?.contiguous()?;
        let down_proj_t = down_proj.t()?.contiguous()?;

        let layer = GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            post_attention_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            attention: GpuAttentionWeights::Full(make_test_attn_weights(
                num_heads,
                num_kv_heads,
                head_dim,
                hidden,
                &device,
            )?),
            mlp: GpuFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
            },
        };

        let cfg = make_test_config(num_heads, num_kv_heads, head_dim, hidden);
        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let backend = test_backend(&device);
        let out = transformer_block(
            &backend,
            &x,
            &layer,
            &cfg,
            &positions,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            &inv_freq,
            1e-6,
            None,
            0,
            None,
        )?;
        assert_eq!(out.dims(), &[batch, seq_len, hidden]);

        Ok(())
    }

    #[test]
    fn test_transformer_block_residual_connections() -> Result<()> {
        // With residual connections, output should differ from zero even with small weights
        let device = Device::Cpu;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let hidden = num_heads * head_dim;
        let intermediate = hidden * 2;

        // Input with known non-zero values
        let x = Tensor::ones((1, 2, hidden), DType::F32, &device)?;
        let positions = vec![0u32, 1];

        let gate_proj = Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?;
        let up_proj = Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?;
        let down_proj = Tensor::randn(0.0_f32, 0.02, (hidden, intermediate), &device)?;
        let gate_proj_t = gate_proj.t()?.contiguous()?;
        let up_proj_t = up_proj.t()?.contiguous()?;
        let down_proj_t = down_proj.t()?.contiguous()?;

        let layer = GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            post_attention_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            attention: GpuAttentionWeights::Full(make_test_attn_weights(
                num_heads,
                num_kv_heads,
                head_dim,
                hidden,
                &device,
            )?),
            mlp: GpuFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
            },
        };

        let cfg = make_test_config(num_heads, num_kv_heads, head_dim, hidden);
        let inv_freq = compute_rotary_inv_freq(head_dim, 10_000.0, &device)?;
        let backend = test_backend(&device);
        let out = transformer_block(
            &backend,
            &x,
            &layer,
            &cfg,
            &positions,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            &inv_freq,
            1e-6,
            None,
            0,
            None,
        )?;

        // Output should not be zero (residual adds input through)
        let vals = out.flatten_all()?.to_vec1::<f32>()?;
        let sum: f32 = vals.iter().map(|v| v.abs()).sum();
        assert!(
            sum > 0.1,
            "residual connections should keep output non-zero, got sum={sum}"
        );
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "output should be finite"
        );

        Ok(())
    }

    #[test]
    fn test_transformer_block_rejects_linear_attention() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 8;

        let layer = GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            post_attention_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            attention: GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                in_proj_qkv: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_z: Tensor::zeros((1, 1), DType::F32, &device)?,
                out_proj: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_a: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_b: Tensor::zeros((1, 1), DType::F32, &device)?,
                conv1d: Tensor::zeros((1, 1), DType::F32, &device)?,
                norm: Tensor::zeros((1,), DType::F32, &device)?,
                a_log: Tensor::zeros((1,), DType::F32, &device)?,
                a_log_gates: Tensor::zeros((1,), DType::F32, &device)?,
                dt_bias: Tensor::zeros((1,), DType::F32, &device)?,
                in_proj_qkv_t: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_z_t: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_a_t: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_b_t: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_ab_t: None,
                out_proj_t: Tensor::zeros((1, 1), DType::F32, &device)?,
                out_proj_marlin: None,
            }),
            mlp: GpuFfnWeights {
                gate_proj: Tensor::zeros((1, hidden), DType::F32, &device)?,
                up_proj: Tensor::zeros((1, hidden), DType::F32, &device)?,
                down_proj: Tensor::zeros((hidden, 1), DType::F32, &device)?,
                gate_proj_t: Tensor::zeros((hidden, 1), DType::F32, &device)?,
                up_proj_t: Tensor::zeros((hidden, 1), DType::F32, &device)?,
                down_proj_t: Tensor::zeros((1, hidden), DType::F32, &device)?,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
            },
        };

        let x = Tensor::ones((1, 1, hidden), DType::F32, &device)?;
        let cfg = make_test_config(2, 1, 4, 8);
        let inv_freq = compute_rotary_inv_freq(4, 10_000.0, &device)?;
        let backend = test_backend(&device);
        let result = transformer_block(
            &backend,
            &x,
            &layer,
            &cfg,
            &[0],
            2,
            1,
            4,
            4,
            &inv_freq,
            1e-6,
            None,
            0,
            None,
        );
        assert!(result.is_err(), "should reject linear attention layers");

        Ok(())
    }

    #[test]
    fn test_weight_to_tensor_f32() -> Result<()> {
        let device = Device::Cpu;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let wt = WeightTensor {
            data: crate::weights::WeightData::owned(bytes),
            shape: vec![2, 3],
            dtype: TensorDType::F32,
            source: None,
        };

        let t = weight_to_tensor(&wt, &device)?;
        assert_eq!(t.dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);

        let vals = t.to_vec2::<f32>()?;
        assert!((vals[0][0] - 1.0).abs() < 1e-6);
        assert!((vals[1][2] - 6.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_weight_to_transposed_tensor_2d_f32_matches_cached_transpose() -> Result<()> {
        let device = Device::Cpu;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let wt = WeightTensor {
            data: crate::weights::WeightData::owned(bytes),
            shape: vec![2, 3],
            dtype: TensorDType::F32,
            source: None,
        };

        let direct = weight_to_transposed_tensor_2d(&wt, &device)?;
        let baseline = cached_transpose(&weight_to_tensor(&wt, &device)?)?;

        assert!(direct.is_contiguous());
        assert_eq!(direct.dims(), &[3, 2]);
        assert_eq!(direct.to_vec2::<f32>()?, baseline.to_vec2::<f32>()?);
        assert_eq!(
            direct.to_vec2::<f32>()?,
            vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
        );
        Ok(())
    }

    #[test]
    fn test_transposed_weight_bytes_2d_preserves_two_byte_elements() -> Result<()> {
        let values: Vec<u16> = vec![1, 2, 3, 4, 5, 6];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let wt = WeightTensor {
            data: crate::weights::WeightData::owned(bytes),
            shape: vec![2, 3],
            dtype: TensorDType::BF16,
            source: None,
        };

        let (transposed, shape) = transposed_weight_bytes_2d(&wt)?;
        let got: Vec<u16> = transposed
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        assert_eq!(shape, [3, 2]);
        assert_eq!(got, vec![1, 4, 2, 5, 3, 6]);
        Ok(())
    }

    #[test]
    fn test_transposed_weight_bytes_2d_parallel_preserves_two_byte_elements() -> Result<()> {
        let rows = 513usize;
        let cols = 1025usize;
        let values: Vec<u16> = (0..rows * cols).map(|idx| idx as u16).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(bytes.len() >= PARALLEL_TRANSPOSE_MIN_BYTES);
        let wt = WeightTensor {
            data: crate::weights::WeightData::owned(bytes),
            shape: vec![rows, cols],
            dtype: TensorDType::BF16,
            source: None,
        };

        let (transposed, shape) = transposed_weight_bytes_2d(&wt)?;

        assert_eq!(shape, [cols, rows]);
        for col in 0..cols {
            for row in 0..rows {
                let got_offset = (col * rows + row) * 2;
                let got = u16::from_le_bytes([transposed[got_offset], transposed[got_offset + 1]]);
                assert_eq!(got, values[row * cols + col]);
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_weight_to_transposed_tensor_2d_metal_matches_cpu_cached_transpose() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        let data: Vec<f32> = vec![1.0, -2.0, 3.5, 4.25, 5.0, -6.75];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let wt = WeightTensor {
            data: crate::weights::WeightData::owned(bytes),
            shape: vec![2, 3],
            dtype: TensorDType::F32,
            source: None,
        };

        let direct = weight_to_transposed_tensor_2d(&wt, &device)?.to_device(&cpu)?;
        let baseline = cached_transpose(&weight_to_tensor(&wt, &cpu)?)?;

        assert!(direct.is_contiguous());
        assert_eq!(direct.dims(), &[3, 2]);
        assert_eq!(direct.to_vec2::<f32>()?, baseline.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn test_cached_transpose_materializes_on_cpu() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;

        let tt = cached_transpose(&t)?;

        assert!(tt.is_contiguous());
        assert_eq!(tt.dims(), &[3, 2]);
        assert_eq!(
            tt.to_vec2::<f32>()?,
            vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
        );
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_cached_transpose_materializes_on_metal() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };
        let t = Tensor::new(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;

        let tt = cached_transpose(&t)?;

        assert!(tt.is_contiguous());
        assert_eq!(tt.dims(), &[3, 2]);
        assert_eq!(
            tt.to_vec2::<f32>()?,
            vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
        );
        Ok(())
    }

    /// Helper: build tiny GpuWeights for testing model_forward shape propagation.
    /// Uses full-attention layers only (no linear attention) with small dimensions.
    fn make_tiny_gpu_weights(
        device: &Device,
        vocab_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        num_layers: usize,
    ) -> Result<GpuWeights> {
        let randn = |shape: &[usize]| -> Result<Tensor> {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.01).sin()) * 0.1).collect();
            Ok(Tensor::new(data, device)?.reshape(shape)?)
        };

        let embed_tokens = randn(&[vocab_size, hidden_size])?;
        let embed_tokens_t = embed_tokens.t()?.contiguous()?;
        let final_norm = Tensor::zeros(hidden_size, DType::F32, device)?;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let q_proj = randn(&[num_heads * head_dim, hidden_size])?;
            let k_proj = randn(&[num_kv_heads * head_dim, hidden_size])?;
            let v_proj = randn(&[num_kv_heads * head_dim, hidden_size])?;
            let o_proj = randn(&[hidden_size, num_heads * head_dim])?;
            let q_proj_t = q_proj.t()?.contiguous()?;
            let k_proj_t = k_proj.t()?.contiguous()?;
            let v_proj_t = v_proj.t()?.contiguous()?;
            let o_proj_t = o_proj.t()?.contiguous()?;
            let gate_proj = randn(&[intermediate_size, hidden_size])?;
            let up_proj = randn(&[intermediate_size, hidden_size])?;
            let down_proj = randn(&[hidden_size, intermediate_size])?;
            let gate_proj_t = gate_proj.t()?.contiguous()?;
            let up_proj_t = up_proj.t()?.contiguous()?;
            let down_proj_t = down_proj.t()?.contiguous()?;
            layers.push(GpuLayerWeights {
                input_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
                post_attention_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
                attention: GpuAttentionWeights::Full(GpuFullAttentionWeights {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                    k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                    q_proj_t,
                    k_proj_t,
                    v_proj_t,
                    qkv_proj_t: None,
                    o_proj_t,
                    q_proj_marlin: None,
                }),
                mlp: GpuFfnWeights {
                    gate_proj,
                    up_proj,
                    down_proj,
                    gate_proj_t,
                    up_proj_t,
                    down_proj_t,
                    gate_proj_marlin: None,
                    up_proj_marlin: None,
                    down_proj_marlin: None,
                },
            });
        }

        // Tests using this helper all set `partial_rotary_factor = 1.0` and
        // `rope_theta = 10000.0`, so rotate every head_dim with base 10k.
        let rotary_inv_freq = compute_rotary_inv_freq(head_dim, 10000.0, device)?;

        Ok(GpuWeights {
            embed_tokens,
            embed_tokens_t,
            layers,
            final_norm,
            rotary_inv_freq,
            mtp: None,
        })
    }

    #[test]
    fn test_model_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let vocab_size = 32;
        let hidden_size = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_size = 32;
        let num_layers = 2;

        let weights = make_tiny_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            num_layers,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: num_layers,
            full_attention_interval: 1, // every layer is full attention
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let token_ids: Vec<u32> = vec![1, 5, 3, 10];
        let backend = test_backend(&device);
        let logits = model_forward(&backend, &token_ids, &weights, &config, None, None, None)?;

        // Expected shape: [1, seq_len, vocab_size]
        assert_eq!(logits.dims(), &[1, 4, vocab_size]);

        Ok(())
    }

    #[test]
    fn test_model_forward_single_token() -> Result<()> {
        let device = Device::Cpu;
        let vocab_size = 16;
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let intermediate_size = 16;

        let weights = make_tiny_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            1, // single layer
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers: 1,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let backend = test_backend(&device);
        let logits = model_forward(&backend, &[7], &weights, &config, None, None, None)?;
        assert_eq!(logits.dims(), &[1, 1, vocab_size]);

        // Logits should be finite
        let vals = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "all logits should be finite"
        );

        Ok(())
    }

    #[test]
    fn test_model_forward_kv_cache_equivalence() -> Result<()> {
        // Verify that model_forward with KV cache produces the same last-position
        // logits as without KV cache, for a multi-token sequence processed
        // incrementally (prefill + decode steps).
        use crate::kv_cache::KvCache;

        let device = Device::Cpu;
        let vocab_size = 16;
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let intermediate_size = 16;
        let num_layers = 2;

        let weights = make_tiny_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            num_layers,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: num_layers,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let tokens: Vec<u32> = vec![1, 5, 3, 10, 7];
        let backend = test_backend(&device);

        // Reference: full forward pass without KV cache
        let logits_ref = model_forward(&backend, &tokens, &weights, &config, None, None, None)?;
        // Extract last position logits: [1, 5, vocab] -> last position
        let last_ref = logits_ref.narrow(1, tokens.len() - 1, 1)?; // [1, 1, vocab]
        let last_ref_vals = last_ref.flatten_all()?.to_vec1::<f32>()?;

        // With KV cache: prefill first 4 tokens, then decode the 5th
        let mut kv_cache =
            KvCache::new(num_layers, num_kv_heads, head_dim, 32, DType::F32, &device)?;

        // Prefill
        let _prefill_logits = model_forward(
            &backend,
            &tokens[..4],
            &weights,
            &config,
            Some(&mut kv_cache),
            None,
            None,
        )?;
        kv_cache.advance(4);
        assert_eq!(kv_cache.seq_len(), 4);

        // Decode the 5th token
        let decode_logits = model_forward(
            &backend,
            &tokens[4..],
            &weights,
            &config,
            Some(&mut kv_cache),
            None,
            None,
        )?;
        kv_cache.advance(1);
        assert_eq!(kv_cache.seq_len(), 5);

        let last_cached_vals = decode_logits
            .narrow(1, 0, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Compare: should be identical (within floating point tolerance)
        assert_eq!(last_ref_vals.len(), last_cached_vals.len());
        for (i, (r, c)) in last_ref_vals.iter().zip(&last_cached_vals).enumerate() {
            assert!(
                (r - c).abs() < 1e-4,
                "logit {i} differs: ref={r}, cached={c}, diff={}",
                (r - c).abs()
            );
        }

        Ok(())
    }

    #[test]
    fn test_model_forward_kv_cache_token_by_token() -> Result<()> {
        // Verify that processing tokens one-by-one with KV cache matches
        // processing all at once without cache.
        use crate::kv_cache::KvCache;

        let device = Device::Cpu;
        let vocab_size = 16;
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let intermediate_size = 16;

        let weights = make_tiny_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            1,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers: 1,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let tokens: Vec<u32> = vec![3, 7, 1];
        let backend = test_backend(&device);

        // Reference
        let logits_ref = model_forward(&backend, &tokens, &weights, &config, None, None, None)?;
        let last_ref = logits_ref
            .narrow(1, 2, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // KV cache: process token by token
        let mut kv_cache = KvCache::new(1, num_kv_heads, head_dim, 16, DType::F32, &device)?;

        // Token 0
        let _ = model_forward(
            &backend,
            &[3],
            &weights,
            &config,
            Some(&mut kv_cache),
            None,
            None,
        )?;
        kv_cache.advance(1);

        // Token 1
        let _ = model_forward(
            &backend,
            &[7],
            &weights,
            &config,
            Some(&mut kv_cache),
            None,
            None,
        )?;
        kv_cache.advance(1);

        // Token 2
        let logits_cached = model_forward(
            &backend,
            &[1],
            &weights,
            &config,
            Some(&mut kv_cache),
            None,
            None,
        )?;
        kv_cache.advance(1);

        let last_cached = logits_cached
            .narrow(1, 0, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        for (i, (r, c)) in last_ref.iter().zip(&last_cached).enumerate() {
            assert!(
                (r - c).abs() < 1e-4,
                "logit {i} differs: ref={r}, cached={c}",
            );
        }

        Ok(())
    }

    /// Helper: build tiny GpuWeights with a mix of full and linear attention layers.
    fn make_hybrid_gpu_weights(
        device: &Device,
        vocab_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        num_layers: usize,
        full_attention_interval: usize,
    ) -> Result<GpuWeights> {
        let randn = |shape: &[usize]| -> Result<Tensor> {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.01).sin()) * 0.1).collect();
            Ok(Tensor::new(data, device)?.reshape(shape)?)
        };

        let embed_tokens = randn(&[vocab_size, hidden_size])?;
        let embed_tokens_t = embed_tokens.t()?.contiguous()?;
        let final_norm = Tensor::zeros(hidden_size, DType::F32, device)?;

        // For linear attention: nk heads with key_head_dim, nv heads with value_head_dim
        // Use same dims as full attention for simplicity
        let nk = num_kv_heads;
        let nv = num_heads;
        let dk = head_dim;
        let dv = head_dim;
        let qkv_dim = nk * dk + nk * dk + nv * dv; // Q + K + V fused
        let conv_kernel = 4;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let is_full = (i + 1) % full_attention_interval == 0;
            let attention = if is_full {
                let q_proj = randn(&[num_heads * head_dim, hidden_size])?;
                let k_proj = randn(&[num_kv_heads * head_dim, hidden_size])?;
                let v_proj = randn(&[num_kv_heads * head_dim, hidden_size])?;
                let o_proj = randn(&[hidden_size, num_heads * head_dim])?;
                let q_proj_t = q_proj.t()?.contiguous()?;
                let k_proj_t = k_proj.t()?.contiguous()?;
                let v_proj_t = v_proj.t()?.contiguous()?;
                let o_proj_t = o_proj.t()?.contiguous()?;
                GpuAttentionWeights::Full(GpuFullAttentionWeights {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                    k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                    q_proj_t,
                    k_proj_t,
                    v_proj_t,
                    qkv_proj_t: None,
                    o_proj_t,
                    q_proj_marlin: None,
                })
            } else {
                let in_proj_qkv = randn(&[qkv_dim, hidden_size])?;
                let in_proj_z = randn(&[nv * dv, hidden_size])?;
                let out_proj = randn(&[hidden_size, nv * dv])?;
                let in_proj_a = randn(&[nv, hidden_size])?;
                let in_proj_b = randn(&[nv, hidden_size])?;
                let in_proj_qkv_t = in_proj_qkv.t()?.contiguous()?;
                let in_proj_z_t = in_proj_z.t()?.contiguous()?;
                let in_proj_a_t = in_proj_a.t()?.contiguous()?;
                let in_proj_b_t = in_proj_b.t()?.contiguous()?;
                let out_proj_t = out_proj.t()?.contiguous()?;
                GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                    in_proj_qkv,
                    in_proj_z,
                    out_proj,
                    in_proj_a,
                    in_proj_b,
                    conv1d: randn(&[qkv_dim, 1, conv_kernel])?,
                    norm: Tensor::ones(dk, DType::F32, device)?,
                    a_log: Tensor::zeros(nv, DType::F32, device)?,
                    a_log_gates: Tensor::zeros(nv, DType::F32, device)?,
                    dt_bias: Tensor::zeros(nv, DType::F32, device)?,
                    in_proj_qkv_t,
                    in_proj_z_t,
                    in_proj_a_t,
                    in_proj_b_t,
                    in_proj_ab_t: None,
                    out_proj_t,
                    out_proj_marlin: None,
                })
            };

            let gate_proj = randn(&[intermediate_size, hidden_size])?;
            let up_proj = randn(&[intermediate_size, hidden_size])?;
            let down_proj = randn(&[hidden_size, intermediate_size])?;
            let gate_proj_t = gate_proj.t()?.contiguous()?;
            let up_proj_t = up_proj.t()?.contiguous()?;
            let down_proj_t = down_proj.t()?.contiguous()?;
            layers.push(GpuLayerWeights {
                input_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
                post_attention_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
                attention,
                mlp: GpuFfnWeights {
                    gate_proj,
                    up_proj,
                    down_proj,
                    gate_proj_t,
                    up_proj_t,
                    down_proj_t,
                    gate_proj_marlin: None,
                    up_proj_marlin: None,
                    down_proj_marlin: None,
                },
            });
        }

        // Tests using this helper set `partial_rotary_factor = 1.0` and
        // `rope_theta = 10000.0`, so rotary_dim = head_dim with base 10k.
        let rotary_inv_freq = compute_rotary_inv_freq(head_dim, 10000.0, device)?;

        Ok(GpuWeights {
            embed_tokens,
            embed_tokens_t,
            layers,
            final_norm,
            rotary_inv_freq,
            mtp: None,
        })
    }

    #[test]
    fn test_model_forward_hybrid_layers() -> Result<()> {
        // Test model_forward with a mix of full and linear (GDN) attention layers
        let device = Device::Cpu;
        let vocab_size = 32;
        let hidden_size = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_size = 32;
        let num_layers = 4;
        let full_attention_interval = 4; // layer 3 is full, layers 0,1,2 are linear

        let weights = make_hybrid_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            num_layers,
            full_attention_interval,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval,
            attn_output_gate: false,
            linear_num_key_heads: num_kv_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let mut linear_state = LinearAttentionState::new(&config, &device)?;

        // Prefill with multiple tokens
        let token_ids: Vec<u32> = vec![1, 5, 3, 10];
        let backend = test_backend(&device);
        let logits = model_forward(
            &backend,
            &token_ids,
            &weights,
            &config,
            None,
            Some(&mut linear_state),
            None,
        )?;
        assert_eq!(logits.dims(), &[1, 4, vocab_size]);

        // All values should be finite (no NaN/Inf)
        let flat = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "logits contain non-finite values"
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    struct ParityScenario {
        label: &'static str,
        vocab_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        num_layers: usize,
        full_attention_interval: usize,
        token_ids: Vec<u32>,
        max_abs_diff: f32,
    }

    /// Runs `model_forward` on CPU and Metal with matching random-weight
    /// models and asserts the logits agree within `scenario.max_abs_diff`.
    /// Drives both parity tests below; the scenario controls whether the
    /// `MetalBackend` SDPA path activates (head_dim ∈ whitelist) or whether
    /// the portable candle fallback runs.
    ///
    /// Returns `Ok(())` without running if Metal isn't available so the
    /// suite stays portable on Linux + CUDA hosts.
    #[cfg(feature = "metal")]
    fn run_cpu_metal_parity(scenario: ParityScenario) -> Result<()> {
        let Some(metal_device) = crate::backend::metal::try_new_metal() else {
            eprintln!("skipping parity test '{}'", scenario.label);
            return Ok(());
        };
        let cpu_device = Device::Cpu;

        let weights_cpu = make_hybrid_gpu_weights(
            &cpu_device,
            scenario.vocab_size,
            scenario.hidden_size,
            scenario.num_heads,
            scenario.num_kv_heads,
            scenario.head_dim,
            scenario.intermediate_size,
            scenario.num_layers,
            scenario.full_attention_interval,
        )?;
        let weights_metal = make_hybrid_gpu_weights(
            &metal_device,
            scenario.vocab_size,
            scenario.hidden_size,
            scenario.num_heads,
            scenario.num_kv_heads,
            scenario.head_dim,
            scenario.intermediate_size,
            scenario.num_layers,
            scenario.full_attention_interval,
        )?;

        // Linear attention dims are 0 when full_attention_interval == 1 (no
        // GDN layers in the model); otherwise set to head_dim so GDN state
        // is shaped for the fallback path.
        let has_linear_layers = scenario.full_attention_interval > 1;
        let linear_num_kv_heads = if has_linear_layers {
            scenario.num_kv_heads
        } else {
            0
        };
        let linear_num_value_heads = if has_linear_layers {
            scenario.num_heads
        } else {
            0
        };
        let linear_head_dim = if has_linear_layers {
            scenario.head_dim
        } else {
            0
        };
        let linear_conv_kernel_dim = if has_linear_layers { 4 } else { 0 };

        let config = kiln_core::config::ModelConfig {
            hidden_size: scenario.hidden_size,
            num_layers: scenario.num_layers,
            num_attention_heads: scenario.num_heads,
            num_kv_heads: scenario.num_kv_heads,
            head_dim: scenario.head_dim,
            intermediate_size: scenario.intermediate_size,
            vocab_size: scenario.vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: if has_linear_layers {
                1
            } else {
                scenario.num_layers
            },
            full_attention_interval: scenario.full_attention_interval,
            attn_output_gate: false,
            linear_num_key_heads: linear_num_kv_heads,
            linear_key_head_dim: linear_head_dim,
            linear_num_value_heads,
            linear_value_head_dim: linear_head_dim,
            linear_conv_kernel_dim,
            partial_rotary_factor: 1.0,
        };

        let cpu_backend = test_backend(&cpu_device);
        let mut cpu_linear = LinearAttentionState::new(&config, &cpu_device)?;
        let logits_cpu = model_forward(
            &cpu_backend,
            &scenario.token_ids,
            &weights_cpu,
            &config,
            None,
            Some(&mut cpu_linear),
            None,
        )?;

        let metal_backend = crate::backend::for_device(&metal_device);
        let mut metal_linear = LinearAttentionState::new(&config, &metal_device)?;
        let logits_metal = model_forward(
            &*metal_backend,
            &scenario.token_ids,
            &weights_metal,
            &config,
            None,
            Some(&mut metal_linear),
            None,
        )?;

        assert_eq!(logits_cpu.dims(), logits_metal.dims());

        let cpu_flat = logits_cpu.flatten_all()?.to_vec1::<f32>()?;
        let metal_flat = logits_metal
            .to_device(&cpu_device)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert!(
            cpu_flat.iter().all(|v| v.is_finite()),
            "{}: CPU logits non-finite",
            scenario.label
        );
        assert!(
            metal_flat.iter().all(|v| v.is_finite()),
            "{}: Metal logits non-finite",
            scenario.label
        );

        let max_abs_diff = cpu_flat
            .iter()
            .zip(metal_flat.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < scenario.max_abs_diff,
            "{}: CPU vs Metal logits diverge: max abs diff = {max_abs_diff} (bound {})",
            scenario.label,
            scenario.max_abs_diff,
        );
        Ok(())
    }

    /// Qwen-shaped: GQA ratio 4, head_dim 128, full attention only. Exercises
    /// `MetalBackend::flash_attn_prefill` (candle SDPA) directly — head_dim
    /// 128 is in the SDPA whitelist, seq_len 12 > 8 for the full SDPA kernel
    /// (not the vector path).
    #[cfg(feature = "metal")]
    #[test]
    fn test_model_forward_parity_sdpa_path() -> Result<()> {
        run_cpu_metal_parity(ParityScenario {
            label: "sdpa_path",
            vocab_size: 32,
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 128,
            hidden_size: 512,
            intermediate_size: 1024,
            num_layers: 2,
            full_attention_interval: 1,
            token_ids: (0..12u32).collect(),
            // SDPA internally accumulates at FP32 but softmax rounds differently
            // from the naive CPU path. 1e-2 accommodates M1 drift; tighten if
            // later hardware proves it's conservative.
            max_abs_diff: 1e-2,
        })
    }

    /// Hybrid full + GDN layers with head_dim 4, below the SDPA whitelist.
    /// `MetalBackend` declines into the portable fallback, so this validates
    /// that the whole candle composition (embed, RMSNorm, RoPE, SwiGLU, naive
    /// softmax+matmul, GDN recurrent loop) runs correctly on Apple Silicon.
    #[cfg(feature = "metal")]
    #[test]
    fn test_model_forward_parity_cpu_vs_metal() -> Result<()> {
        run_cpu_metal_parity(ParityScenario {
            label: "portable_fallback",
            vocab_size: 32,
            hidden_size: 16,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            num_layers: 4,
            full_attention_interval: 4,
            token_ids: vec![1, 5, 3, 10],
            max_abs_diff: 1e-3,
        })
    }

    #[test]
    fn test_model_forward_hybrid_decode() -> Result<()> {
        // Test prefill + decode with linear attention state persistence
        let device = Device::Cpu;
        let vocab_size = 32;
        let hidden_size = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_size = 32;
        let num_layers = 4;
        let full_attention_interval = 4;

        let weights = make_hybrid_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            num_layers,
            full_attention_interval,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval,
            attn_output_gate: false,
            linear_num_key_heads: num_kv_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let mut kv_cache = KvCache::new(1, num_kv_heads, head_dim, 32, DType::F32, &device)?;
        let mut linear_state = LinearAttentionState::new(&config, &device)?;
        let backend = test_backend(&device);

        // Prefill
        let prefill_logits = model_forward(
            &backend,
            &[1, 5, 3],
            &weights,
            &config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            None,
        )?;
        kv_cache.advance(3);
        assert_eq!(prefill_logits.dims(), &[1, 3, vocab_size]);

        // Decode: single token should work with persisted linear state
        let decode_logits = model_forward(
            &backend,
            &[10],
            &weights,
            &config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            None,
        )?;
        kv_cache.advance(1);
        assert_eq!(decode_logits.dims(), &[1, 1, vocab_size]);

        // Both should produce finite values
        let flat = decode_logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "decode logits contain non-finite values"
        );

        Ok(())
    }

    #[test]
    fn test_linear_attention_state_new() -> Result<()> {
        let device = Device::Cpu;
        let config = kiln_core::config::ModelConfig {
            hidden_size: 16,
            num_layers: 4,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            vocab_size: 32,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 4,
            attn_output_gate: false,
            linear_num_key_heads: 2,
            linear_key_head_dim: 4,
            linear_num_value_heads: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let state = LinearAttentionState::new(&config, &device)?;
        // 3 linear layers (layers 0,1,2; layer 3 is full)
        assert_eq!(state.recurrent_states.len(), 3);
        assert_eq!(state.conv_states.len(), 3);
        // Recurrent state shape: [1, nv, dk, dv]
        assert_eq!(state.recurrent_states[0].dims(), &[1, 4, 4, 4]);
        assert_eq!(state.recurrent_states[0].dtype(), DType::F32);
        // Conv state shape: [1, qkv_dim, kernel_size-1] where qkv_dim = 2*(nk*dk) + nv*dv = 2*8+16=32
        let qkv_dim = 2 * (2 * 4) + 4 * 4; // 32
        assert_eq!(state.conv_states[0].dims(), &[1, qkv_dim, 3]);
        assert_eq!(state.conv_states[0].dtype(), DType::F32);

        let batched = LinearAttentionState::new_with_batch(&config, 3, &device)?;
        assert_eq!(batched.recurrent_states.len(), 3);
        assert_eq!(batched.conv_states.len(), 3);
        assert_eq!(batched.recurrent_states[0].dims(), &[3, 4, 4, 4]);
        assert_eq!(batched.recurrent_states[0].dtype(), DType::F32);
        assert_eq!(batched.conv_states[0].dims(), &[3, qkv_dim, 3]);
        assert_eq!(batched.conv_states[0].dtype(), DType::F32);
        assert!(LinearAttentionState::new_with_batch(&config, 0, &device).is_err());

        Ok(())
    }

    #[test]
    fn test_linear_attention_state_batch_row_assembly_and_scatter() -> Result<()> {
        let device = Device::Cpu;
        let config = kiln_core::config::ModelConfig {
            hidden_size: 16,
            num_layers: 4,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            vocab_size: 32,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 4,
            attn_output_gate: false,
            linear_num_key_heads: 2,
            linear_key_head_dim: 4,
            linear_num_value_heads: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let mut row0 = LinearAttentionState::new(&config, &device)?;
        let mut row1 = LinearAttentionState::new(&config, &device)?;
        let recurrent_values0: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let recurrent_values1: Vec<f32> = (0..64).map(|i| 1000.0 + i as f32).collect();
        let conv_values0: Vec<f32> = (0..96).map(|i| 2000.0 + i as f32).collect();
        let conv_values1: Vec<f32> = (0..96).map(|i| 3000.0 + i as f32).collect();
        row0.recurrent_states[0] = Tensor::from_slice(
            &recurrent_values0,
            (1usize, 4usize, 4usize, 4usize),
            &device,
        )?;
        row1.recurrent_states[0] = Tensor::from_slice(
            &recurrent_values1,
            (1usize, 4usize, 4usize, 4usize),
            &device,
        )?;
        row0.conv_states[0] =
            Tensor::from_slice(&conv_values0, (1usize, 32usize, 3usize), &device)?;
        row1.conv_states[0] =
            Tensor::from_slice(&conv_values1, (1usize, 32usize, 3usize), &device)?;

        let batched = LinearAttentionState::from_batch_rows(&[&row0, &row1])?;
        assert_eq!(batched.batch_size()?, 2);
        assert_eq!(batched.recurrent_states[0].dims(), &[2, 4, 4, 4]);
        assert_eq!(batched.conv_states[0].dims(), &[2, 32, 3]);
        assert!(LinearAttentionState::from_batch_rows(&[&batched]).is_err());

        let split = batched.split_batch_rows()?;
        assert_eq!(split.len(), 2);
        assert_eq!(
            split[0].recurrent_states[0]
                .flatten_all()?
                .to_vec1::<f32>()?,
            row0.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            split[1].recurrent_states[0]
                .flatten_all()?
                .to_vec1::<f32>()?,
            row1.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            split[0].conv_states[0].to_vec3::<f32>()?,
            row0.conv_states[0].to_vec3::<f32>()?
        );
        assert_eq!(
            split[1].conv_states[0].to_vec3::<f32>()?,
            row1.conv_states[0].to_vec3::<f32>()?
        );

        let mut dst0 = LinearAttentionState::new(&config, &device)?;
        let mut dst1 = LinearAttentionState::new(&config, &device)?;
        {
            let mut destinations = [&mut dst0, &mut dst1];
            batched.scatter_batch_rows(&mut destinations)?;
        }
        assert_eq!(
            dst0.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?,
            row0.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            dst1.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?,
            row1.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            dst0.conv_states[0].to_vec3::<f32>()?,
            row0.conv_states[0].to_vec3::<f32>()?
        );
        assert_eq!(
            dst1.conv_states[0].to_vec3::<f32>()?,
            row1.conv_states[0].to_vec3::<f32>()?
        );

        let mut one_destination = [&mut dst0];
        assert!(batched.scatter_batch_rows(&mut one_destination).is_err());

        let mut replace_dst0 = LinearAttentionState::new(&config, &device)?;
        let mut replace_dst1 = LinearAttentionState::new(&config, &device)?;
        {
            let mut destinations = [&mut replace_dst0, &mut replace_dst1];
            batched.scatter_batch_rows_replace(&mut destinations)?;
        }
        assert_eq!(
            replace_dst0.recurrent_states[0]
                .flatten_all()?
                .to_vec1::<f32>()?,
            row0.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            replace_dst1.recurrent_states[0]
                .flatten_all()?
                .to_vec1::<f32>()?,
            row1.recurrent_states[0].flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            replace_dst0.conv_states[0].to_vec3::<f32>()?,
            row0.conv_states[0].to_vec3::<f32>()?
        );
        assert_eq!(
            replace_dst1.conv_states[0].to_vec3::<f32>()?,
            row1.conv_states[0].to_vec3::<f32>()?
        );

        let mut one_replace_destination = [&mut replace_dst0];
        assert!(
            batched
                .scatter_batch_rows_replace(&mut one_replace_destination)
                .is_err()
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_linear_attention_state_uses_bf16_on_metal_for_bf16_models() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };

        let config = kiln_core::config::ModelConfig {
            hidden_size: 16,
            num_layers: 4,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            vocab_size: 32,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::BF16,
            num_full_attention_layers: 1,
            full_attention_interval: 4,
            attn_output_gate: false,
            linear_num_key_heads: 2,
            linear_key_head_dim: 4,
            linear_num_value_heads: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let state = LinearAttentionState::new(&config, &device)?;
        assert_eq!(state.recurrent_states[0].dtype(), DType::BF16);
        assert_eq!(state.conv_states[0].dtype(), DType::F32);

        let batched = LinearAttentionState::new_with_batch(&config, 3, &device)?;
        assert_eq!(batched.recurrent_states[0].dims(), &[3, 4, 4, 4]);
        assert_eq!(batched.recurrent_states[0].dtype(), DType::BF16);
        assert_eq!(
            batched.conv_states[0].dims(),
            &[3, config.linear_qkv_dim(), 3]
        );
        assert_eq!(batched.conv_states[0].dtype(), DType::F32);

        let row0 = LinearAttentionState::new(&config, &device)?;
        let row1 = LinearAttentionState::new(&config, &device)?;
        let assembled = LinearAttentionState::from_batch_rows(&[&row0, &row1])?;
        assert_eq!(assembled.batch_size()?, 2);
        assert_eq!(assembled.recurrent_states[0].dims(), &[2, 4, 4, 4]);
        assert_eq!(assembled.recurrent_states[0].dtype(), DType::BF16);
        assert_eq!(
            assembled.conv_states[0].dims(),
            &[2, config.linear_qkv_dim(), 3]
        );
        assert_eq!(assembled.conv_states[0].dtype(), DType::F32);
        let split = assembled.split_batch_rows()?;
        assert_eq!(split.len(), 2);
        assert_eq!(split[0].recurrent_states[0].dims(), &[1, 4, 4, 4]);
        assert_eq!(split[0].recurrent_states[0].dtype(), DType::BF16);
        assert_eq!(
            split[0].conv_states[0].dims(),
            &[1, config.linear_qkv_dim(), 3]
        );
        assert_eq!(split[0].conv_states[0].dtype(), DType::F32);

        Ok(())
    }

    #[test]
    fn test_linear_attention_state_vulkan_inference_backend_uses_model_dtype() -> Result<()> {
        let device = Device::Cpu;
        let mut config = make_test_config(2, 1, 4, 8);

        let default_cpu = LinearAttentionState::new_with_batch_for_inference(&config, 2, &device)?;
        assert_eq!(default_cpu.recurrent_states[0].dims(), &[2, 2, 4, 4]);
        assert_eq!(default_cpu.recurrent_states[0].dtype(), DType::F32);
        assert_eq!(default_cpu.conv_states[0].dtype(), DType::F32);

        let named_cpu = LinearAttentionState::new_with_batch_for_inference_backend(
            &config,
            2,
            &device,
            Some("cpu"),
        )?;
        assert_eq!(named_cpu.recurrent_states[0].dtype(), DType::F32);
        assert_eq!(named_cpu.conv_states[0].dtype(), DType::F32);

        let vulkan = LinearAttentionState::new_with_batch_for_inference_backend(
            &config,
            2,
            &device,
            Some("vulkan"),
        )?;
        assert_eq!(vulkan.recurrent_states[0].dims(), &[2, 2, 4, 4]);
        assert_eq!(vulkan.recurrent_states[0].dtype(), DType::BF16);
        assert_eq!(vulkan.conv_states[0].dtype(), DType::F32);

        config.dtype = kiln_core::config::DType::FP16;
        let vulkan_fp16 = LinearAttentionState::new_with_batch_for_inference_backend(
            &config,
            2,
            &device,
            Some("vulkan"),
        )?;
        assert_eq!(vulkan_fp16.recurrent_states[0].dtype(), DType::F16);
        assert_eq!(vulkan_fp16.conv_states[0].dtype(), DType::F32);

        Ok(())
    }

    #[test]
    fn test_causal_mask_with_offset() -> Result<()> {
        let device = Device::Cpu;
        // Simulate decode: 1 new query, 4 total KV (3 cached + 1 new)
        let scores = Tensor::ones((1, 1, 1, 4), DType::F32, &device)?;
        let masked = apply_causal_mask_with_offset(&scores, 1, 4, 3)?;
        // Single query should attend to all 4 positions (no masking for q_len=1)
        let vals = masked.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            vals.iter().all(|v| (*v - 1.0).abs() < 1e-6),
            "single query token should attend to all KV positions"
        );

        // Simulate prefill with offset: 2 new queries, 5 total KV (3 cached + 2 new)
        let scores = Tensor::ones((1, 1, 2, 5), DType::F32, &device)?;
        let masked = apply_causal_mask_with_offset(&scores, 2, 5, 3)?;
        let vals = masked.flatten_all()?.to_vec1::<f32>()?;
        // Row 0 (abs pos 3): can attend to positions 0..4 (first 4), mask position 4
        assert!((vals[0] - 1.0).abs() < 1e-6); // pos 0: ok
        assert!((vals[1] - 1.0).abs() < 1e-6); // pos 1: ok
        assert!((vals[2] - 1.0).abs() < 1e-6); // pos 2: ok
        assert!((vals[3] - 1.0).abs() < 1e-6); // pos 3 (self): ok
        assert!(vals[4].is_infinite() && vals[4] < 0.0); // pos 4: masked
        // Row 1 (abs pos 4): can attend to all 5 positions
        assert!(vals[5..10].iter().all(|v| (*v - 1.0).abs() < 1e-6));

        Ok(())
    }

    // ------------------------------------------------------------------
    // GDN chunkwise correctness test (Phase 6)
    // ------------------------------------------------------------------

    /// Reference per-token GDN recurrence, mirroring the pre-Phase-6 loop
    /// that used to live in `gated_deltanet_forward`. Kept in the test
    /// module (never called from production) so the chunkwise implementation
    /// can be cross-checked against the arithmetically simple form.
    ///
    /// Inputs are already transposed to [B, nv, T, *]; state is [B, nv, dk, dv].
    fn gdn_sequential_reference(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        state: &mut Tensor,
    ) -> Result<Tensor> {
        let (_, _, seq_len, _) = q.dims4()?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let q_t = q.narrow(2, t, 1)?; // [B, nv, 1, dk]
            let k_t = k.narrow(2, t, 1)?; // [B, nv, 1, dk]
            let v_t = v.narrow(2, t, 1)?.squeeze(2)?; // [B, nv, dv]
            let beta_t = beta.narrow(2, t, 1)?.squeeze(2)?; // [B, nv]
            let g_t = g.narrow(2, t, 1)?.squeeze(2)?; // [B, nv]

            let g_exp = g_t.exp()?.unsqueeze(2)?.unsqueeze(3)?; // [B, nv, 1, 1]
            *state = state.broadcast_mul(&g_exp)?;

            let kv_mem = k_t.matmul(&*state)?.squeeze(2)?; // [B, nv, dv]
            let delta: Tensor = (v_t - kv_mem)?.broadcast_mul(&beta_t.unsqueeze(2)?)?; // [B, nv, dv]

            let k_col = k_t.squeeze(2)?.unsqueeze(3)?; // [B, nv, dk, 1]
            let outer = k_col.broadcast_mul(&delta.unsqueeze(2)?)?; // [B, nv, dk, dv]
            *state = (&*state + &outer)?;

            let out_t = q_t.matmul(&*state)?; // [B, nv, 1, dv]
            outputs.push(out_t);
        }
        Ok(Tensor::cat(&outputs, 2)?)
    }

    /// Deterministic tensor of the given shape filled with values from a
    /// simple hash of the index. Avoids depending on candle's RNG (which
    /// uses process-global state) and keeps the test reproducible.
    fn det_tensor(shape: &[usize], scale: f32, bias: f32, device: &Device) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| {
                // Cheap mixable pseudo-random: stretch i through two sin
                // waves of different frequencies. Gives values in roughly
                // [-1, 1] with no exact repeats for small n.
                let x = (i as f32 * 0.7283).sin() + (i as f32 * 1.3719).cos();
                (x * 0.5) * scale + bias
            })
            .collect();
        Ok(Tensor::from_vec(data, shape, device)?)
    }

    #[test]
    fn test_gdn_chunkwise_matches_sequential() -> Result<()> {
        // Small, fully-on-CPU shapes. We use F32 here so the comparison
        // is against the same numerical path the chunkwise form takes
        // for its decay cumulative products; the task spec's bf16
        // tolerance (<1e-3) is comfortably satisfied in F32 as well.
        let device = Device::Cpu;
        let dtype = DType::F32;

        let b = 1;
        let nv = 2;
        let t = 8;
        let dk = 4;
        let dv = 4;
        let chunk_size = 4;

        let q = det_tensor(&[b, nv, t, dk], 1.0, 0.0, &device)?.to_dtype(dtype)?;
        let k = det_tensor(&[b, nv, t, dk], 1.0, 0.0, &device)?.to_dtype(dtype)?;
        let v = det_tensor(&[b, nv, t, dv], 1.0, 0.0, &device)?.to_dtype(dtype)?;
        // beta ∈ (0, 1): pass through sigmoid-like shift.
        let beta_raw = det_tensor(&[b, nv, t], 2.0, 0.0, &device)?.to_dtype(dtype)?;
        let beta = {
            let ones = Tensor::ones_like(&beta_raw)?;
            (&ones / (&ones + &beta_raw.neg()?.exp()?)?)?
        };
        // g ∈ (-0.2, 0): small negative decays so cumulative sum stays sane.
        let g_raw = det_tensor(&[b, nv, t], 0.2, 0.0, &device)?.to_dtype(dtype)?;
        let g = (g_raw.abs()? * (-1.0_f64))?;

        let state_init = Tensor::zeros((b, nv, dk, dv), dtype, &device)?;
        let backend = test_backend(&device);

        let mut state_chunk = state_init.clone();
        let out_chunk = gdn_chunkwise_recurrence(
            &backend,
            &q,
            &k,
            &v,
            &beta,
            &g,
            &mut state_chunk,
            chunk_size,
        )?;

        let mut state_seq = state_init.clone();
        let out_seq = gdn_sequential_reference(&q, &k, &v, &beta, &g, &mut state_seq)?;

        let out_diff = (&out_chunk - &out_seq)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let state_diff = (&state_chunk - &state_seq)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;

        // Task acceptance: max abs diff < 1e-3 in bf16. We run the test in
        // F32 so the actual tolerance is much tighter; guard against both
        // silent divergence and silent upgrade of the bf16 tolerance bound.
        assert!(
            out_diff < 1e-3,
            "chunkwise vs sequential output diff too large: {out_diff}",
        );
        assert!(
            state_diff < 1e-3,
            "chunkwise vs sequential state diff too large: {state_diff}",
        );

        // Also test chunk_size >= seq_len (single-chunk path) and
        // chunk_size == 1 (decode-like path) for coverage.
        for &cs in &[1usize, t] {
            let mut state_a = state_init.clone();
            let out_a =
                gdn_chunkwise_recurrence(&backend, &q, &k, &v, &beta, &g, &mut state_a, cs)?;
            let mut state_b = state_init.clone();
            let out_b = gdn_sequential_reference(&q, &k, &v, &beta, &g, &mut state_b)?;
            let d = (&out_a - &out_b)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            let sd = (&state_a - &state_b)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(d < 1e-3, "chunkwise(cs={cs}) output diff {d}");
            assert!(sd < 1e-3, "chunkwise(cs={cs}) state diff {sd}");
        }

        Ok(())
    }

    #[test]
    fn test_gdn_recurrent_backward_no_grad_matches_autograd_cpu() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let b = 1;
        let nv = 2;
        let t = 5;
        let dk = 3;
        let dv = 4;
        let chunk_size = 4;

        let q = det_tensor(&[b, nv, t, dk], 0.35, 0.02, &device)?.to_dtype(dtype)?;
        let k = det_tensor(&[b, nv, t, dk], 0.30, -0.01, &device)?.to_dtype(dtype)?;
        let v = det_tensor(&[b, nv, t, dv], 0.25, 0.03, &device)?.to_dtype(dtype)?;
        let beta_raw = det_tensor(&[b, nv, t], 1.0, 0.0, &device)?.to_dtype(dtype)?;
        let beta = {
            let ones = Tensor::ones_like(&beta_raw)?;
            (&ones / (&ones + &beta_raw.neg()?.exp()?)?)?
        };
        let g = det_tensor(&[b, nv, t], 0.08, -0.12, &device)?
            .to_dtype(dtype)?
            .neg()?
            .abs()?
            .neg()?;
        let state = det_tensor(&[b, nv, dk, dv], 0.10, 0.01, &device)?.to_dtype(dtype)?;
        let upstream = det_tensor(&[b, nv, t, dv], 0.20, -0.02, &device)?.to_dtype(dtype)?;
        let grad_exit_state = det_tensor(&[b, nv, dk, dv], 0.15, 0.04, &device)?.to_dtype(dtype)?;
        let backend = test_backend(&device);

        let q_var = Var::from_tensor(&q)?;
        let k_var = Var::from_tensor(&k)?;
        let v_var = Var::from_tensor(&v)?;
        let beta_var = Var::from_tensor(&beta)?;
        let g_var = Var::from_tensor(&g)?;
        let state_var = Var::from_tensor(&state)?;
        let mut state_for_autograd = state_var.as_tensor().clone();
        let out = gdn_chunkwise_recurrence(
            &backend,
            q_var.as_tensor(),
            k_var.as_tensor(),
            v_var.as_tensor(),
            beta_var.as_tensor(),
            g_var.as_tensor(),
            &mut state_for_autograd,
            chunk_size,
        )?;
        let out_term = (&out.to_dtype(DType::F32)? * &upstream)?.sum_all()?;
        let state_term =
            (&state_for_autograd.to_dtype(DType::F32)? * &grad_exit_state)?.sum_all()?;
        let loss = (&out_term + &state_term)?;
        let grads = loss.backward()?;

        let manual = gdn_recurrent_backward_no_grad(
            &backend,
            &q,
            &k,
            &v,
            &beta,
            &g,
            &state,
            &upstream,
            Some(&grad_exit_state),
            chunk_size,
        )?;

        fn assert_grad_close(
            name: &str,
            actual: &Tensor,
            expected: &Tensor,
            tol: f32,
        ) -> Result<()> {
            let diff = (actual - expected)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(
                diff < tol,
                "{name} gradient diff too large: {diff} >= {tol}"
            );
            Ok(())
        }

        assert_grad_close(
            "q",
            &manual.dq,
            grads.get(q_var.as_tensor()).context("missing q grad")?,
            1e-4,
        )?;
        assert_grad_close(
            "k",
            &manual.dk,
            grads.get(k_var.as_tensor()).context("missing k grad")?,
            1e-4,
        )?;
        assert_grad_close(
            "v",
            &manual.dv,
            grads.get(v_var.as_tensor()).context("missing v grad")?,
            1e-4,
        )?;
        assert_grad_close(
            "beta",
            &manual.dbeta,
            grads
                .get(beta_var.as_tensor())
                .context("missing beta grad")?,
            1e-4,
        )?;
        assert_grad_close(
            "g",
            &manual.dg,
            grads.get(g_var.as_tensor()).context("missing g grad")?,
            1e-4,
        )?;
        assert_grad_close(
            "state",
            manual.d_state.as_ref().context("missing state grad")?,
            grads
                .get(state_var.as_tensor())
                .context("missing state grad")?,
            1e-4,
        )?;

        Ok(())
    }

    #[test]
    fn test_gdn_chunkwise_masks_decay_before_exp() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let b = 1;
        let nv = 2;
        let t = 13;
        let dk = 4;
        let dv = 4;

        let q = det_tensor(&[b, nv, t, dk], 0.3, 0.0, &device)?.to_dtype(dtype)?;
        let k = det_tensor(&[b, nv, t, dk], 0.2, 0.0, &device)?.to_dtype(dtype)?;
        let v = det_tensor(&[b, nv, t, dv], 0.4, 0.0, &device)?.to_dtype(dtype)?;
        let beta = Tensor::ones((b, nv, t), dtype, &device)?;
        let g = Tensor::from_vec(vec![-100.0f32; b * nv * t], (b, nv, t), &device)?;
        let state_init = Tensor::zeros((b, nv, dk, dv), dtype, &device)?;
        let backend = test_backend(&device);

        let mut state = state_init.clone();
        let out = gdn_chunkwise_recurrence(&backend, &q, &k, &v, &beta, &g, &mut state, t)?;

        for (name, tensor) in [("out", &out), ("state", &state)] {
            let values = tensor.flatten_all()?.to_vec1::<f32>()?;
            assert!(
                values.iter().all(|v| v.is_finite()),
                "{name} contains non-finite values"
            );
        }

        Ok(())
    }

    #[test]
    fn test_gdn_single_token_matches_sequential() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let b = 1;
        let nv = 2;
        let t = 1;
        let dk = 4;
        let dv = 4;

        let q = det_tensor(&[b, nv, t, dk], 1.0, 0.0, &device)?.to_dtype(dtype)?;
        let k = det_tensor(&[b, nv, t, dk], 0.8, 0.1, &device)?.to_dtype(dtype)?;
        let v = det_tensor(&[b, nv, t, dv], 0.6, -0.2, &device)?.to_dtype(dtype)?;
        let beta_raw = det_tensor(&[b, nv, t], 1.5, 0.0, &device)?.to_dtype(dtype)?;
        let beta = {
            let ones = Tensor::ones_like(&beta_raw)?;
            (&ones / (&ones + &beta_raw.neg()?.exp()?)?)?
        };
        let g_raw = det_tensor(&[b, nv, t], 0.2, 0.0, &device)?.to_dtype(dtype)?;
        let g = (g_raw.abs()? * (-1.0_f64))?;

        let state_init = det_tensor(&[b, nv, dk, dv], 0.1, 0.0, &device)?.to_dtype(dtype)?;
        let backend = test_backend(&device);

        let mut state_fast = state_init.clone();
        let out_fast = gdn_chunkwise_recurrence(
            &backend,
            &q,
            &k,
            &v,
            &beta,
            &g,
            &mut state_fast,
            GDN_CHUNK_SIZE,
        )?;

        let mut state_seq = state_init.clone();
        let out_seq = gdn_sequential_reference(&q, &k, &v, &beta, &g, &mut state_seq)?;

        let out_diff = (&out_fast - &out_seq)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let state_diff = (&state_fast - &state_seq)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;

        assert!(
            out_diff < 1e-5,
            "single-token fast path output drifted: max_abs_diff={out_diff:e}"
        );
        assert!(
            state_diff < 1e-5,
            "single-token fast path state drifted: max_abs_diff={state_diff:e}"
        );
        Ok(())
    }

    /// Correctness test for the vendored kiln-gdn-kernel CUDA fused
    /// forward-substitution kernel.
    ///
    /// Compares the fused kernel output against the per-token candle
    /// fallback on the same random bf16 inputs at kiln's exact GDN config
    /// (B=1, nv=32, C=64, dv=128). Asserts max abs diff < 1e-2 and mean
    /// abs diff < 1e-3 — the fused path uses F32 accumulators and
    /// per-token bf16 round-trips, so finite-precision drift is bounded
    /// by bf16 rounding noise.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gdn_kernel_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available, skipping test_gdn_kernel_matches_fallback");
                return Ok(());
            }
        };

        let b = 1usize;
        let nv = 32usize;
        let c = 64usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64);

        let n_a = b * nv * c * c;
        let n_v = b * nv * c * dv;
        let n_b = b * nv * c;

        let a_data: Vec<f32> = (0..n_a)
            .map(|_| rng.random_range(-0.05f32..0.05f32))
            .collect();
        let v_data: Vec<f32> = (0..n_v)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let beta_data: Vec<f32> = (0..n_b).map(|_| rng.random_range(0.5f32..1.5f32)).collect();

        let a_f32 = Tensor::from_slice(&a_data, (b, nv, c, c), &device)?;
        let v_f32 = Tensor::from_slice(&v_data, (b, nv, c, dv), &device)?;
        let beta_f32 = Tensor::from_slice(&beta_data, (b, nv, c), &device)?;

        // Make A_strict actually strictly lower triangular (matches what
        // the recurrence produces upstream of compute_w_chunk).
        let mask = strict_lower_tri_mask(c, DType::F32, &device)?;
        let a_f32 = a_f32.broadcast_mul(&mask)?;

        let a = a_f32.to_dtype(DType::BF16)?;
        let v = v_f32.to_dtype(DType::BF16)?;
        let beta = beta_f32.to_dtype(DType::BF16)?;

        let backend = crate::backend::for_device(&device);
        let w_kernel = compute_w_chunk(&*backend, &a, &v, &beta, c)?; // CUDA kernel
        let w_fb = compute_w_chunk_fallback(&a, &v, &beta, c)?; // candle per-token

        let diff = (w_kernel.to_dtype(DType::F32)? - w_fb.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

        eprintln!("gdn-kernel vs fallback: max_abs_diff={max:e}, mean_abs_diff={mean:e}");

        assert!(
            max < 1e-2,
            "kernel output exceeds tolerance: max_abs_diff = {max:e}"
        );
        assert!(
            mean < 1e-3,
            "kernel mean drift exceeds tolerance: mean_abs_diff = {mean:e}"
        );

        Ok(())
    }

    /// Correctness test for the Metal fused forward-substitution kernel.
    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_gdn_forward_substitution_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!(
                "Metal not available, skipping test_metal_gdn_forward_substitution_matches_fallback"
            );
            return Ok(());
        };

        let b = 1usize;
        let nv = 8usize;
        let c = 16usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0xFACE_FEED_u64);

        let n_a = b * nv * c * c;
        let n_v = b * nv * c * dv;
        let n_b = b * nv * c;

        let a_data: Vec<f32> = (0..n_a)
            .map(|_| rng.random_range(-0.05f32..0.05f32))
            .collect();
        let v_data: Vec<f32> = (0..n_v)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let beta_data: Vec<f32> = (0..n_b).map(|_| rng.random_range(0.5f32..1.5f32)).collect();

        let a_f32 = Tensor::from_slice(&a_data, (b, nv, c, c), &device)?;
        let v_f32 = Tensor::from_slice(&v_data, (b, nv, c, dv), &device)?;
        let beta_f32 = Tensor::from_slice(&beta_data, (b, nv, c), &device)?;

        let mask = strict_lower_tri_mask(c, DType::F32, &device)?;
        let a_f32 = a_f32.broadcast_mul(&mask)?;

        let a = a_f32.to_dtype(DType::BF16)?;
        let v = v_f32.to_dtype(DType::BF16)?;
        let beta = beta_f32.to_dtype(DType::BF16)?;

        let backend = crate::backend::for_device(&device);
        let w_kernel = compute_w_chunk(&*backend, &a, &v, &beta, c)?;
        let w_fb = compute_w_chunk_fallback(&a, &v, &beta, c)?;

        let diff = (w_kernel.to_dtype(DType::F32)? - w_fb.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

        eprintln!(
            "metal gdn-forward-sub vs fallback: max_abs_diff={max:e}, mean_abs_diff={mean:e}"
        );

        assert!(
            max < 1e-2,
            "metal forward-sub kernel output exceeds tolerance: max_abs_diff = {max:e}"
        );
        assert!(
            mean < 1e-3,
            "metal forward-sub kernel mean drift exceeds tolerance: mean_abs_diff = {mean:e}"
        );

        Ok(())
    }

    /// Parity check for the single-token recurrent CUDA kernel.
    ///
    /// Compares output and final state of `gdn_chunkwise_recurrence` with
    /// the new fused recurrent kernel against `gdn_sequential_reference`
    /// at kiln's exact GDN config (B=1, nv=32, dk=128, dv=128, T=1).
    /// Tolerance matches the chunkwise CUDA kernel test.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gdn_recurrent_kernel_matches_reference() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "CUDA not available, skipping test_gdn_recurrent_kernel_matches_reference"
                );
                return Ok(());
            }
        };

        let b = 1usize;
        let nv = 32usize;
        let t = 1usize;
        let dk = 128usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0xDECAFBADu64);

        let n_qk = b * nv * t * dk;
        let n_v = b * nv * t * dv;
        let n_b = b * nv * t;
        let n_s = b * nv * dk * dv;

        let q_data: Vec<f32> = (0..n_qk)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let k_data: Vec<f32> = (0..n_qk)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let v_data: Vec<f32> = (0..n_v)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let beta_data: Vec<f32> = (0..n_b).map(|_| rng.random_range(0.3f32..1.2f32)).collect();
        // Small negative gates so exp(g) stays in (~0.8, 1.0).
        let g_data: Vec<f32> = (0..n_b)
            .map(|_| rng.random_range(-0.2f32..0.0f32))
            .collect();
        let s_data: Vec<f32> = (0..n_s)
            .map(|_| rng.random_range(-0.1f32..0.1f32))
            .collect();

        let q_f32 = Tensor::from_slice(&q_data, (b, nv, t, dk), &device)?;
        let k_f32 = Tensor::from_slice(&k_data, (b, nv, t, dk), &device)?;
        let v_f32 = Tensor::from_slice(&v_data, (b, nv, t, dv), &device)?;
        let beta_f32 = Tensor::from_slice(&beta_data, (b, nv, t), &device)?;
        let g_f32 = Tensor::from_slice(&g_data, (b, nv, t), &device)?;
        let state_f32 = Tensor::from_slice(&s_data, (b, nv, dk, dv), &device)?;

        let q = q_f32.to_dtype(DType::BF16)?;
        let k = k_f32.to_dtype(DType::BF16)?;
        let v = v_f32.to_dtype(DType::BF16)?;
        let beta = beta_f32.to_dtype(DType::BF16)?;
        let g = g_f32.to_dtype(DType::BF16)?;
        let state_bf16 = state_f32.to_dtype(DType::BF16)?;

        // Reference path: F32 sequential recurrence on the same numerical
        // inputs (cast back to F32 from the bf16 round-trip so the bf16
        // quantization is shared between the two paths and only the kernel
        // arithmetic differs).
        let q_ref = q.to_dtype(DType::F32)?;
        let k_ref = k.to_dtype(DType::F32)?;
        let v_ref = v.to_dtype(DType::F32)?;
        let beta_ref = beta.to_dtype(DType::F32)?;
        let g_ref = g.to_dtype(DType::F32)?;
        let mut state_ref = state_bf16.to_dtype(DType::F32)?;
        let out_ref =
            gdn_sequential_reference(&q_ref, &k_ref, &v_ref, &beta_ref, &g_ref, &mut state_ref)?;

        // Kernel path: chunkwise dispatcher with seq_len == 1 routes to
        // the new fused recurrent kernel. Make sure no prior test left the
        // kill-switch set in this process.
        // SAFETY: cargo test is single-threaded per test by default and we
        // are only mutating an env var that the dispatcher reads at the top
        // of the same call below. No other thread observes it concurrently.
        unsafe {
            std::env::remove_var("KILN_DISABLE_GDN_KERNEL");
        }
        let backend = crate::backend::for_device(&device);
        let mut state_kernel = state_bf16.clone();
        let out_kernel =
            gdn_chunkwise_recurrence(&*backend, &q, &k, &v, &beta, &g, &mut state_kernel, 1)?;

        let out_diff = (out_kernel.to_dtype(DType::F32)? - &out_ref)?;
        let abs = out_diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        let s_diff = (state_kernel.to_dtype(DType::F32)? - &state_ref)?;
        let s_abs = s_diff.abs()?;
        let s_max = s_abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let s_mean = s_abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

        eprintln!(
            "gdn-recurrent vs reference: out max={max:e} mean={mean:e}, state max={s_max:e} mean={s_mean:e}"
        );

        assert!(
            max < 1e-2,
            "recurrent kernel output exceeds tolerance: max_abs_diff = {max:e}"
        );
        assert!(
            mean < 1e-3,
            "recurrent kernel mean drift exceeds tolerance: mean_abs_diff = {mean:e}"
        );
        assert!(
            s_max < 1e-2,
            "recurrent kernel state exceeds tolerance: max_abs_diff = {s_max:e}"
        );
        assert!(
            s_mean < 1e-3,
            "recurrent kernel state mean drift exceeds tolerance: mean_abs_diff = {s_mean:e}"
        );

        Ok(())
    }

    /// Parity check for the single-token recurrent Metal kernel.
    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_gdn_recurrent_kernel_matches_reference() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!(
                "Metal not available, skipping test_metal_gdn_recurrent_kernel_matches_reference"
            );
            return Ok(());
        };

        let b = 1usize;
        let nv = 16usize;
        let t = 1usize;
        let dk = 128usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0xBEEFu64);

        let n_qk = b * nv * t * dk;
        let n_v = b * nv * t * dv;
        let n_b = b * nv * t;
        let n_s = b * nv * dk * dv;

        let q_data: Vec<f32> = (0..n_qk)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let k_data: Vec<f32> = (0..n_qk)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let v_data: Vec<f32> = (0..n_v)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let beta_data: Vec<f32> = (0..n_b).map(|_| rng.random_range(0.3f32..1.2f32)).collect();
        let g_data: Vec<f32> = (0..n_b)
            .map(|_| rng.random_range(-0.2f32..0.0f32))
            .collect();
        let s_data: Vec<f32> = (0..n_s)
            .map(|_| rng.random_range(-0.1f32..0.1f32))
            .collect();

        let q = Tensor::from_slice(&q_data, (b, nv, t, dk), &device)?.to_dtype(DType::BF16)?;
        let k = Tensor::from_slice(&k_data, (b, nv, t, dk), &device)?.to_dtype(DType::BF16)?;
        let v = Tensor::from_slice(&v_data, (b, nv, t, dv), &device)?.to_dtype(DType::BF16)?;
        let beta = Tensor::from_slice(&beta_data, (b, nv, t), &device)?.to_dtype(DType::BF16)?;
        let g = Tensor::from_slice(&g_data, (b, nv, t), &device)?.to_dtype(DType::BF16)?;
        let state_bf16 =
            Tensor::from_slice(&s_data, (b, nv, dk, dv), &device)?.to_dtype(DType::BF16)?;

        let q_ref = q.to_dtype(DType::F32)?;
        let k_ref = k.to_dtype(DType::F32)?;
        let v_ref = v.to_dtype(DType::F32)?;
        let beta_ref = beta.to_dtype(DType::F32)?;
        let g_ref = g.to_dtype(DType::F32)?;
        let mut state_ref = state_bf16.to_dtype(DType::F32)?;
        let out_ref =
            gdn_sequential_reference(&q_ref, &k_ref, &v_ref, &beta_ref, &g_ref, &mut state_ref)?;

        let backend = crate::backend::for_device(&device);
        if !backend.supports_gdn_recurrent_step() {
            eprintln!("Metal recurrent kernel disabled, skipping parity test");
            return Ok(());
        }
        let mut state_kernel = state_bf16.clone();
        let out_kernel =
            gdn_chunkwise_recurrence(&*backend, &q, &k, &v, &beta, &g, &mut state_kernel, 1)?;

        let out_diff = (out_kernel.to_dtype(DType::F32)? - &out_ref)?;
        let abs = out_diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        let s_diff = (state_kernel.to_dtype(DType::F32)? - &state_ref)?;
        let s_abs = s_diff.abs()?;
        let s_max = s_abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let s_mean = s_abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

        eprintln!(
            "metal gdn-recurrent vs reference: out max={max:e} mean={mean:e}, state max={s_max:e} mean={s_mean:e}"
        );

        assert!(
            max < 1e-2,
            "metal recurrent kernel output exceeds tolerance: max_abs_diff = {max:e}"
        );
        assert!(
            mean < 1e-3,
            "metal recurrent kernel mean drift exceeds tolerance: mean_abs_diff = {mean:e}"
        );
        assert!(
            s_max < 1e-2,
            "metal recurrent kernel state exceeds tolerance: max_abs_diff = {s_max:e}"
        );
        assert!(
            s_mean < 1e-3,
            "metal recurrent kernel state mean drift exceeds tolerance: mean_abs_diff = {s_mean:e}"
        );

        Ok(())
    }

    /// Parity check for the fused chunk-prep CUDA kernel.
    ///
    /// Generates random bf16 `kkt`, `qkt`, `ks_entry`, `q_s`, `v`, `g` at
    /// kiln's GDN prefill shape (B=1, nv=32, C=64, dv=128), then asserts
    /// that the fused `gdn_chunk_prep` kernel produces the same six
    /// output tensors as the candle-op reference chain it replaces.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gdn_chunk_prep_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available, skipping test_gdn_chunk_prep_matches_fallback");
                return Ok(());
            }
        };

        let b = 1usize;
        let nv = 32usize;
        let c = 64usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0xB00B1E5_u64);

        let n_g = b * nv * c;
        let n_v = b * nv * c * dv;
        let n_cc = b * nv * c * c;

        // Small negative gates so big_g stays in a reasonable range — the
        // recurrence produces g_t near zero so the cumulative sum caps
        // around -10 at most.
        let g_data: Vec<f32> = (0..n_g)
            .map(|_| rng.random_range(-0.15f32..0.0f32))
            .collect();
        let v_data: Vec<f32> = (0..n_v)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let kkt_data: Vec<f32> = (0..n_cc)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let qkt_data: Vec<f32> = (0..n_cc)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let ks_data: Vec<f32> = (0..n_v)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let qs_data: Vec<f32> = (0..n_v)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();

        let g = Tensor::from_slice(&g_data, (b, nv, c), &device)?.to_dtype(DType::BF16)?;
        let v = Tensor::from_slice(&v_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let kkt = Tensor::from_slice(&kkt_data, (b, nv, c, c), &device)?.to_dtype(DType::BF16)?;
        let qkt = Tensor::from_slice(&qkt_data, (b, nv, c, c), &device)?.to_dtype(DType::BF16)?;
        let ks_entry =
            Tensor::from_slice(&ks_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let q_s = Tensor::from_slice(&qs_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;

        // Kernel path.
        let (a_strict_k, b_mask_k, v_prime_k, q_s_scaled_k, decay_last_col_k, p_last_k) =
            kiln_gdn_kernel::gdn_chunk_prep(&g, &v, &kkt, &qkt, &ks_entry, &q_s)?;

        // Candle reference chain — mirrors the else branch in
        // gdn_chunkwise_recurrence.
        let g_f32 = g.to_dtype(DType::F32)?;
        let big_g = g_f32.cumsum(candle_core::D::Minus1)?; // [B, nv, C] F32
        let big_g_col = big_g.unsqueeze(3)?;
        let big_g_row = big_g.unsqueeze(2)?;
        let decay_f32 = big_g_col.broadcast_sub(&big_g_row)?.exp()?;
        let decay = decay_f32.to_dtype(DType::BF16)?;
        let p = big_g.exp()?.to_dtype(DType::BF16)?;
        let p_col = p.unsqueeze(3)?;

        let strict_mask = strict_lower_tri_mask(c, DType::BF16, &device)?;
        let causal_mask = causal_lower_tri_mask(c, DType::BF16, &device)?;

        let v_prime_ref = (&v - ks_entry.broadcast_mul(&p_col)?)?;
        let a_strict_ref = kkt
            .broadcast_mul(&decay)?
            .broadcast_mul(&strict_mask)?
            .contiguous()?;
        let b_mask_ref = qkt
            .broadcast_mul(&decay)?
            .broadcast_mul(&causal_mask)?
            .contiguous()?;
        let q_s_scaled_ref = q_s.broadcast_mul(&p_col)?;

        let g_last = big_g.narrow(2, c - 1, 1)?; // [B, nv, 1]
        let decay_last_col_ref = g_last.broadcast_sub(&big_g)?.exp()?.to_dtype(DType::BF16)?; // [B, nv, C]
        let p_last_ref = g_last.squeeze(2)?.exp()?.to_dtype(DType::BF16)?; // [B, nv]

        let check = |name: &str, k: &Tensor, r: &Tensor| -> Result<()> {
            let diff = (k.to_dtype(DType::F32)? - r.to_dtype(DType::F32)?)?;
            let abs = diff.abs()?;
            let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
            eprintln!("gdn-chunk-prep {name}: max={max:e} mean={mean:e}");
            assert!(
                max < 1e-2,
                "chunk-prep {name} max_abs_diff {max:e} exceeds 1e-2"
            );
            assert!(
                mean < 1e-3,
                "chunk-prep {name} mean_abs_diff {mean:e} exceeds 1e-3"
            );
            Ok(())
        };

        check("a_strict", &a_strict_k, &a_strict_ref)?;
        check("b_mask", &b_mask_k, &b_mask_ref)?;
        check("v_prime", &v_prime_k, &v_prime_ref)?;
        check("q_s_scaled", &q_s_scaled_k, &q_s_scaled_ref)?;
        check("decay_last_col", &decay_last_col_k, &decay_last_col_ref)?;
        check("p_last", &p_last_k, &p_last_ref)?;

        Ok(())
    }

    /// Parity check for the fused post-prep prefill chunk body.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gdn_chunk_body_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available, skipping test_gdn_chunk_body_matches_fallback");
                return Ok(());
            }
        };

        let b = 1usize;
        let nv = 32usize;
        let c = 64usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0xFACE1234_u64);
        let n_cc = b * nv * c * c;
        let n_cdv = b * nv * c * dv;
        let n_c = b * nv * c;

        let a_data: Vec<f32> = (0..n_cc)
            .map(|idx| {
                let t = (idx / c) % c;
                let i = idx % c;
                if i < t {
                    rng.random_range(-0.15f32..0.15f32)
                } else {
                    0.0
                }
            })
            .collect();
        let b_data: Vec<f32> = (0..n_cc)
            .map(|_| rng.random_range(-0.2f32..0.2f32))
            .collect();
        let v_data: Vec<f32> = (0..n_cdv)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let qss_data: Vec<f32> = (0..n_cdv)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let beta_data: Vec<f32> = (0..n_c).map(|_| rng.random_range(0.3f32..1.1f32)).collect();
        let decay_data: Vec<f32> = (0..n_c).map(|_| rng.random_range(0.6f32..1.0f32)).collect();

        let a_strict =
            Tensor::from_slice(&a_data, (b, nv, c, c), &device)?.to_dtype(DType::BF16)?;
        let b_mask = Tensor::from_slice(&b_data, (b, nv, c, c), &device)?.to_dtype(DType::BF16)?;
        let v_prime =
            Tensor::from_slice(&v_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let q_s_scaled =
            Tensor::from_slice(&qss_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let beta = Tensor::from_slice(&beta_data, (b, nv, c), &device)?.to_dtype(DType::BF16)?;
        let decay_last_col =
            Tensor::from_slice(&decay_data, (b, nv, c), &device)?.to_dtype(DType::BF16)?;

        let (out_kernel, ww_kernel) = kiln_gdn_kernel::gdn_chunk_scan(
            &a_strict,
            &b_mask,
            &v_prime,
            &q_s_scaled,
            &beta,
            &decay_last_col,
        )?;

        let (out_ref, ww_ref) = compute_chunk_body_reference(
            &a_strict,
            &b_mask,
            &v_prime,
            &q_s_scaled,
            &beta,
            &decay_last_col.unsqueeze(3)?,
        )?;

        let check = |name: &str, got: &Tensor, want: &Tensor| -> Result<()> {
            let diff = (got.to_dtype(DType::F32)? - want.to_dtype(DType::F32)?)?;
            let abs = diff.abs()?;
            let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
            eprintln!("gdn-chunk-body {name}: max={max:e} mean={mean:e}");
            assert!(
                max < 2e-2,
                "chunk-body {name} max_abs_diff {max:e} exceeds 2e-2"
            );
            assert!(
                mean < 2e-3,
                "chunk-body {name} mean_abs_diff {mean:e} exceeds 2e-3"
            );
            Ok(())
        };

        check("out_chunk", &out_kernel, &out_ref)?;
        check("w_weighted", &ww_kernel, &ww_ref)?;
        Ok(())
    }

    /// Parity check for the fused full-chunk CUDA prefill path.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gdn_full_chunk_forward_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "CUDA not available, skipping test_gdn_full_chunk_forward_matches_fallback"
                );
                return Ok(());
            }
        };

        let b = 1usize;
        let nv = 32usize;
        let c = 64usize;
        let dk = 128usize;
        let dv = 128usize;

        let mut rng = StdRng::seed_from_u64(0x5EED_CAFE_u64);
        let n_c = b * nv * c;
        let n_cdv = b * nv * c * dv;
        let n_cc = b * nv * c * c;
        let n_dkc = b * nv * dk * c;
        let n_dkdv = b * nv * dk * dv;

        let g_data: Vec<f32> = (0..n_c)
            .map(|_| rng.random_range(-0.15f32..0.0f32))
            .collect();
        let v_data: Vec<f32> = (0..n_cdv)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let kkt_data: Vec<f32> = (0..n_cc)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let qkt_data: Vec<f32> = (0..n_cc)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let ks_data: Vec<f32> = (0..n_cdv)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let qs_data: Vec<f32> = (0..n_cdv)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let beta_data: Vec<f32> = (0..n_c).map(|_| rng.random_range(0.3f32..1.1f32)).collect();
        let kt_data: Vec<f32> = (0..n_dkc)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let state_data: Vec<f32> = (0..n_dkdv)
            .map(|_| rng.random_range(-0.25f32..0.25f32))
            .collect();

        let g = Tensor::from_slice(&g_data, (b, nv, c), &device)?.to_dtype(DType::BF16)?;
        let v = Tensor::from_slice(&v_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let kkt = Tensor::from_slice(&kkt_data, (b, nv, c, c), &device)?.to_dtype(DType::BF16)?;
        let qkt = Tensor::from_slice(&qkt_data, (b, nv, c, c), &device)?.to_dtype(DType::BF16)?;
        let ks_entry =
            Tensor::from_slice(&ks_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let q_s = Tensor::from_slice(&qs_data, (b, nv, c, dv), &device)?.to_dtype(DType::BF16)?;
        let beta = Tensor::from_slice(&beta_data, (b, nv, c), &device)?.to_dtype(DType::BF16)?;
        let k_t = Tensor::from_slice(&kt_data, (b, nv, dk, c), &device)?.to_dtype(DType::BF16)?;
        let mut state_kernel =
            Tensor::from_slice(&state_data, (b, nv, dk, dv), &device)?.to_dtype(DType::BF16)?;
        let state_ref = state_kernel.clone();

        let out_kernel = kiln_gdn_kernel::gdn_full_chunk_forward(
            &g,
            &v,
            &kkt,
            &qkt,
            &ks_entry,
            &q_s,
            &beta,
            &k_t,
            &mut state_kernel,
        )?;

        let (a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last) =
            kiln_gdn_kernel::gdn_chunk_prep(&g, &v, &kkt, &qkt, &ks_entry, &q_s)?;
        let (out_ref, ww_ref) = compute_chunk_body_reference(
            &a_strict,
            &b_mask,
            &v_prime,
            &q_s_scaled,
            &beta,
            &decay_last_col.unsqueeze(3)?,
        )?;
        let p_last_u = p_last.unsqueeze(2)?.unsqueeze(3)?;
        let state_expected =
            (state_ref.broadcast_mul(&p_last_u)? + k_t.matmul(&ww_ref)?)?.contiguous()?;

        let check =
            |name: &str, got: &Tensor, want: &Tensor, max_tol: f32, mean_tol: f32| -> Result<()> {
                let diff = (got.to_dtype(DType::F32)? - want.to_dtype(DType::F32)?)?;
                let abs = diff.abs()?;
                let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
                eprintln!("gdn-full-chunk {name}: max={max:e} mean={mean:e}");
                assert!(
                    max < max_tol,
                    "full-chunk {name} max_abs_diff {max:e} exceeds {max_tol:e}"
                );
                assert!(
                    mean < mean_tol,
                    "full-chunk {name} mean_abs_diff {mean:e} exceeds {mean_tol:e}"
                );
                Ok(())
            };

        check("out_chunk", &out_kernel, &out_ref, 2e-2, 2e-3)?;
        check("state", &state_kernel, &state_expected, 3.5e-2, 4e-3)?;
        Ok(())
    }

    /// Parity check for the fused causal_conv1d_update kernel against the
    /// portable `causal_conv1d_decode` + `cuda_silu` chain, at Qwen3.5-4B's
    /// exact decode shape: B=1, C=linear_qkv_dim=8192, K=4.
    ///
    /// Verifies (a) the silu-fused F32 output matches within bf16-rounding
    /// noise and (b) the mutated conv_state matches bit-for-bit (both paths
    /// write the same K-1 previous inputs from the same bf16 source).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_causal_conv1d_update_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "CUDA not available, skipping test_causal_conv1d_update_matches_fallback"
                );
                return Ok(());
            }
        };

        let batch = 1usize;
        let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
        let kernel_size = 4usize;

        let mut rng = StdRng::seed_from_u64(0xC0_1DBEEF);
        let n_x = batch * channels * 1;
        let n_w = channels * kernel_size;
        let n_s = batch * channels * (kernel_size - 1);

        let x_data: Vec<f32> = (0..n_x)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let w_data: Vec<f32> = (0..n_w)
            .map(|_| rng.random_range(-0.1f32..0.1f32))
            .collect();
        let s_data: Vec<f32> = (0..n_s)
            .map(|_| rng.random_range(-0.3f32..0.3f32))
            .collect();

        let x_f32 = Tensor::from_slice(&x_data, (batch, channels, 1), &device)?;
        let w_f32 = Tensor::from_slice(&w_data, (channels, 1, kernel_size), &device)?;
        let s_init = Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1), &device)?;

        let x = x_f32.to_dtype(DType::BF16)?;
        let w = w_f32.to_dtype(DType::BF16)?;

        // Fallback path: candle decode + silu in F32.
        let mut s_fb = s_init.clone();
        let out_fb = causal_conv1d_decode(&x, &w, &mut s_fb, kernel_size)?;
        let out_fb = cuda_silu(&out_fb.to_dtype(DType::F32)?)?;

        // Fused kernel path via the backend dispatch.
        let backend = crate::backend::for_device(&device);
        if !backend.supports_causal_conv1d_update() {
            eprintln!(
                "backend declines causal_conv1d_update (KILN_DISABLE_FUSED_CONV1D?); skipping"
            );
            return Ok(());
        }
        let mut s_k = s_init.clone();
        let out_k = match backend.causal_conv1d_update(&x, &w, &mut s_k, kernel_size)? {
            Some(t) => t,
            None => {
                eprintln!("backend declined causal_conv1d_update at Qwen3.5 envelope; skipping");
                return Ok(());
            }
        };

        // Output parity (silu fused on the kernel side).
        let diff = (out_k.to_dtype(DType::F32)? - out_fb.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!("conv1d_update vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
        assert!(
            max < 2e-3,
            "fused conv1d_update output max_abs_diff={max:e} exceeds 2e-3"
        );
        assert!(
            mean < 5e-4,
            "fused conv1d_update output mean_abs_diff={mean:e} exceeds 5e-4"
        );

        // State parity — both paths write the same K-1 previous inputs.
        let sdiff = (s_k.to_dtype(DType::F32)? - s_fb.to_dtype(DType::F32)?)?;
        let smax = sdiff.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        eprintln!("conv1d_update state parity: max_abs_diff={smax:e}");
        assert!(
            smax < 1e-5,
            "fused conv1d_update state max_abs_diff={smax:e} exceeds 1e-5"
        );

        Ok(())
    }

    /// Parity check for the fused CUDA causal_conv1d prefill kernel against
    /// the portable `causal_conv1d_prefill` + `cuda_silu` chain, at the native
    /// MTP draft shape that exercises `seq_len > 1`.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_causal_conv1d_prefill_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "CUDA not available, skipping test_causal_conv1d_prefill_matches_fallback"
                );
                return Ok(());
            }
        };

        let batch = 1usize;
        let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
        let seq_len = 512usize;
        let kernel_size = 4usize;

        let mut rng = StdRng::seed_from_u64(0xC0_1DC0DE);
        let n_x = batch * channels * seq_len;
        let n_w = channels * kernel_size;
        let n_s = batch * channels * (kernel_size - 1);

        let x_data: Vec<f32> = (0..n_x)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let w_data: Vec<f32> = (0..n_w)
            .map(|_| rng.random_range(-0.1f32..0.1f32))
            .collect();
        let s_data: Vec<f32> = (0..n_s)
            .map(|_| rng.random_range(-0.3f32..0.3f32))
            .collect();

        let x = Tensor::from_slice(&x_data, (batch, channels, seq_len), &device)?
            .to_dtype(DType::BF16)?;
        let w = Tensor::from_slice(&w_data, (channels, 1, kernel_size), &device)?
            .to_dtype(DType::BF16)?;
        let s_init = Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1), &device)?;

        let mut s_fb = s_init.clone();
        let out_fb = causal_conv1d_prefill_with_dtype(&x, &w, &mut s_fb, kernel_size, DType::F32)?;
        let out_fb = cuda_silu(&out_fb)?;

        let backend = crate::backend::for_device(&device);
        if !backend.supports_causal_conv1d_prefill() {
            eprintln!(
                "backend declines causal_conv1d_prefill (KILN_DISABLE_FUSED_CONV1D?); skipping"
            );
            return Ok(());
        }
        let mut s_k = s_init.clone();
        let out_k = match backend.causal_conv1d_prefill(&x, &w, &mut s_k, kernel_size)? {
            Some(t) => t,
            None => {
                eprintln!("backend declined causal_conv1d_prefill at Qwen3.5 envelope; skipping");
                return Ok(());
            }
        };

        let diff = (out_k.to_dtype(DType::F32)? - out_fb.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!("conv1d_prefill vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
        assert!(
            max < 2e-3,
            "fused conv1d_prefill output max_abs_diff={max:e} exceeds 2e-3"
        );
        assert!(
            mean < 5e-4,
            "fused conv1d_prefill output mean_abs_diff={mean:e} exceeds 5e-4"
        );

        let sdiff = (s_k.to_dtype(DType::F32)? - s_fb.to_dtype(DType::F32)?)?;
        let smax = sdiff.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        eprintln!("conv1d_prefill state parity: max_abs_diff={smax:e}");
        assert!(
            smax < 1e-5,
            "fused conv1d_prefill state max_abs_diff={smax:e} exceeds 1e-5"
        );

        Ok(())
    }

    /// Metal parity check for `backend.causal_conv1d_update` against the same
    /// portable `causal_conv1d_decode` + `cuda_silu` oracle used by CUDA.
    #[cfg(feature = "metal")]
    #[test]
    fn test_causal_conv1d_update_matches_fallback_metal() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!(
                "Metal unavailable, skipping test_causal_conv1d_update_matches_fallback_metal"
            );
            return Ok(());
        };

        let batch = 1usize;
        let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
        let kernel_size = 4usize;

        let mut rng = StdRng::seed_from_u64(0xC0_1DBEEF);
        let n_x = batch * channels;
        let n_w = channels * kernel_size;
        let n_s = batch * channels * (kernel_size - 1);

        let x_data: Vec<f32> = (0..n_x)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let w_data: Vec<f32> = (0..n_w)
            .map(|_| rng.random_range(-0.1f32..0.1f32))
            .collect();
        let s_data: Vec<f32> = (0..n_s)
            .map(|_| rng.random_range(-0.3f32..0.3f32))
            .collect();

        let x_f32 = Tensor::from_slice(&x_data, (batch, channels, 1), &device)?;
        let w_f32 = Tensor::from_slice(&w_data, (channels, 1, kernel_size), &device)?;
        let s_init = Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1), &device)?;

        let x = x_f32.to_dtype(DType::BF16)?;
        let w = w_f32.to_dtype(DType::BF16)?;

        let mut s_fb = s_init.clone();
        let out_fb = causal_conv1d_decode(&x, &w, &mut s_fb, kernel_size)?;
        let out_fb = cuda_silu(&out_fb.to_dtype(DType::F32)?)?;

        let backend = crate::backend::for_device(&device);
        if !backend.supports_causal_conv1d_update() {
            eprintln!(
                "backend declines causal_conv1d_update (KILN_DISABLE_FUSED_CONV1D?); skipping"
            );
            return Ok(());
        }
        let mut s_k = s_init.clone();
        let out_k = match backend.causal_conv1d_update(&x, &w, &mut s_k, kernel_size)? {
            Some(t) => t,
            None => {
                eprintln!("backend declined causal_conv1d_update at Qwen3.5 envelope; skipping");
                return Ok(());
            }
        };

        let diff = (out_k.to_dtype(DType::F32)? - out_fb.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!("metal conv1d_update vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}");
        assert!(
            max < 2e-3,
            "metal conv1d_update output max_abs_diff={max:e} exceeds 2e-3"
        );
        assert!(
            mean < 5e-4,
            "metal conv1d_update output mean_abs_diff={mean:e} exceeds 5e-4"
        );

        let sdiff = (s_k.to_dtype(DType::F32)? - s_fb.to_dtype(DType::F32)?)?;
        let smax = sdiff.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        eprintln!("metal conv1d_update state parity: max_abs_diff={smax:e}");
        assert!(
            smax < 1e-5,
            "metal conv1d_update state max_abs_diff={smax:e} exceeds 1e-5"
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_causal_conv1d_prefill_bf16_parity_on_metal() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!(
                "Metal not available, skipping test_causal_conv1d_prefill_bf16_parity_on_metal"
            );
            return Ok(());
        };

        let batch = 1usize;
        let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
        let seq_len = 16usize;
        let kernel_size = 4usize;

        let mut rng = StdRng::seed_from_u64(0xBF16_C0DE);
        let n_x = batch * channels * seq_len;
        let n_w = channels * kernel_size;
        let n_s = batch * channels * (kernel_size - 1);

        let x_data: Vec<f32> = (0..n_x)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let w_data: Vec<f32> = (0..n_w)
            .map(|_| rng.random_range(-0.1f32..0.1f32))
            .collect();
        let s_data: Vec<f32> = (0..n_s)
            .map(|_| rng.random_range(-0.3f32..0.3f32))
            .collect();

        let x = Tensor::from_slice(&x_data, (batch, channels, seq_len), &device)?
            .to_dtype(DType::BF16)?;
        let w = Tensor::from_slice(&w_data, (channels, 1, kernel_size), &device)?
            .to_dtype(DType::BF16)?;
        let s_init = Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1), &device)?;

        let mut s_ref = s_init.clone();
        let out_ref =
            causal_conv1d_prefill_with_dtype(&x, &w, &mut s_ref, kernel_size, DType::F32)?;
        let out_ref = cuda_silu(&out_ref)?;

        let mut s_bf16 = s_init.clone();
        assert_eq!(
            causal_conv1d_prefill_compute_dtype(&x, &w, &s_bf16, kernel_size),
            DType::BF16
        );
        let out_bf16 = causal_conv1d_prefill(&x, &w, &mut s_bf16, kernel_size)?;
        assert_eq!(out_bf16.dtype(), DType::BF16);
        assert_eq!(s_bf16.dtype(), DType::F32);
        let out_bf16 = cuda_silu(&out_bf16)?;

        let diff = (out_bf16.to_dtype(DType::F32)? - out_ref.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!("conv1d_prefill bf16 vs f32: max_abs_diff={max:e} mean_abs_diff={mean:e}");
        assert!(
            max < 2e-2,
            "bf16 prefill output max_abs_diff={max:e} exceeds 2e-2"
        );
        assert!(
            mean < 2e-3,
            "bf16 prefill output mean_abs_diff={mean:e} exceeds 2e-3"
        );

        let sdiff = (s_bf16.to_dtype(DType::F32)? - s_ref.to_dtype(DType::F32)?)?;
        let smax = sdiff.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        eprintln!("conv1d_prefill bf16 state parity: max_abs_diff={smax:e}");
        assert!(
            smax < 1e-6,
            "bf16 prefill state max_abs_diff={smax:e} exceeds 1e-6"
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_causal_conv1d_prefill_kernel_matches_fallback() -> Result<()> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let Some(device) = crate::backend::metal::try_new_metal() else {
            eprintln!(
                "Metal not available, skipping test_metal_causal_conv1d_prefill_kernel_matches_fallback"
            );
            return Ok(());
        };

        let batch = 1usize;
        let channels = 8192usize; // Qwen3.5-4B linear_qkv_dim
        let seq_len = 16usize;
        let kernel_size = 4usize;

        let mut rng = StdRng::seed_from_u64(0xC0FFEE_8175);
        let n_x = batch * channels * seq_len;
        let n_w = channels * kernel_size;
        let n_s = batch * channels * (kernel_size - 1);

        let x_data: Vec<f32> = (0..n_x)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let w_data: Vec<f32> = (0..n_w)
            .map(|_| rng.random_range(-0.1f32..0.1f32))
            .collect();
        let s_data: Vec<f32> = (0..n_s)
            .map(|_| rng.random_range(-0.3f32..0.3f32))
            .collect();

        let x = Tensor::from_slice(&x_data, (batch, channels, seq_len), &device)?
            .to_dtype(DType::BF16)?;
        let w = Tensor::from_slice(&w_data, (channels, 1, kernel_size), &device)?
            .to_dtype(DType::BF16)?;
        let s_init = Tensor::from_slice(&s_data, (batch, channels, kernel_size - 1), &device)?;

        let mut s_ref = s_init.clone();
        let out_ref =
            causal_conv1d_prefill_with_dtype(&x, &w, &mut s_ref, kernel_size, DType::F32)?;
        let out_ref = cuda_silu(&out_ref)?;

        let backend = crate::backend::for_device(&device);
        assert!(backend.supports_causal_conv1d_prefill());
        let mut s_kernel = s_init.clone();
        let out_kernel = match backend.causal_conv1d_prefill(&x, &w, &mut s_kernel, kernel_size)? {
            Some(out) => out,
            None => {
                eprintln!("Metal backend declined causal_conv1d_prefill; skipping");
                return Ok(());
            }
        };
        assert_eq!(out_kernel.dtype(), DType::F32);
        assert_eq!(s_kernel.dtype(), DType::F32);

        let diff = (out_kernel.to_dtype(DType::F32)? - out_ref.to_dtype(DType::F32)?)?;
        let abs = diff.abs()?;
        let max = abs.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = abs.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
        eprintln!(
            "metal conv1d_prefill kernel vs fallback: max_abs_diff={max:e} mean_abs_diff={mean:e}"
        );
        assert!(
            max < 1e-5,
            "metal prefill output max_abs_diff={max:e} exceeds 1e-5"
        );
        assert!(
            mean < 1e-6,
            "metal prefill output mean_abs_diff={mean:e} exceeds 1e-6"
        );

        let sdiff = (s_kernel.to_dtype(DType::F32)? - s_ref.to_dtype(DType::F32)?)?;
        let smax = sdiff.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        eprintln!("metal conv1d_prefill kernel state parity: max_abs_diff={smax:e}");
        assert!(
            smax < 1e-6,
            "metal prefill state max_abs_diff={smax:e} exceeds 1e-6"
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Phase 7: streaming/tiled GDN prefill — CPU parity tests.
    //
    // Each test compares the monolithic `model_forward_paged` against
    // `model_forward_paged_streaming_with` running multiple tiles. Both runs
    // start from fresh `LinearAttentionState` + `PagedKvCache` so the
    // recurrent state hand-off and per-tile paged writes are exercised end
    // to end. Tests use `last_token_only=false` so we can compare the full
    // last-tile logits row-by-row against the matching slice of the
    // monolithic logits.
    // -----------------------------------------------------------------------

    /// Shared config for all streaming parity tests. Picks a hybrid layer
    /// stack (3 GDN + 1 full attention with `full_attention_interval=4`,
    /// scaled to 8 layers so we get 6 GDN layers exercising the recurrent
    /// hand-off across tile boundaries).
    fn streaming_test_config() -> kiln_core::config::ModelConfig {
        let num_layers = 8;
        let full_attention_interval = 4; // layers 3, 7 are full → 2 full + 6 linear
        kiln_core::config::ModelConfig {
            hidden_size: 16,
            num_layers,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            vocab_size: 32,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 2,
            full_attention_interval,
            attn_output_gate: false,
            linear_num_key_heads: 2,
            linear_key_head_dim: 4,
            linear_num_value_heads: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        }
    }

    /// Build a paged cache + sequential block table sized for `seq_len` tokens
    /// with `block_size`-token blocks (block_size = GDN_CHUNK_SIZE so block
    /// boundaries coincide with the smallest legal tile boundary).
    fn make_paged_setup(
        config: &kiln_core::config::ModelConfig,
        seq_len: usize,
        block_size: usize,
        device: &Device,
    ) -> Result<(PagedKvCache, BlockTable)> {
        let num_blocks = (seq_len + block_size - 1) / block_size;
        let cache = PagedKvCache::new(
            config.num_full_attention_layers,
            num_blocks,
            block_size,
            config.num_kv_heads,
            config.head_dim,
            DType::F32,
            device,
        )?;
        let mut block_table = BlockTable::new();
        for i in 0..num_blocks as u32 {
            block_table.push(i);
        }
        Ok((cache, block_table))
    }

    /// Deterministic token sequence for parity testing. Stays inside vocab.
    fn deterministic_tokens(seq_len: usize, vocab_size: u32) -> Vec<u32> {
        (0..seq_len)
            .map(|i| ((i as u32 * 13 + 7) % vocab_size).max(1))
            .collect()
    }

    /// Run monolithic vs streaming on the same config + tokens, return
    /// `(monolithic_full_logits[1, T, V], streaming_full_last_tile_logits[1, last_tile_len, V])`
    /// where the streaming pass uses `tile_size` and `last_token_only=false`.
    fn run_parity(
        config: &kiln_core::config::ModelConfig,
        token_ids: &[u32],
        tile_size: usize,
        block_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let device = Device::Cpu;
        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = test_backend(&device);

        // Monolithic: single forward pass, full LM head.
        let (mut mono_cache, mono_bt) =
            make_paged_setup(config, token_ids.len(), block_size, &device)?;
        let mut mono_state = LinearAttentionState::new(config, &device)?;
        let mono_logits = model_forward_paged(
            &backend,
            token_ids,
            &weights,
            config,
            &mut mono_cache,
            &mono_bt,
            0,
            Some(&mut mono_state),
            None,
            None,
        )?;

        // Streaming: tiled prefill with last_token_only=false so the final
        // tile produces a full per-position logits slice we can compare
        // against the matching window of the monolithic output.
        let (mut stream_cache, stream_bt) =
            make_paged_setup(config, token_ids.len(), block_size, &device)?;
        let mut stream_state = LinearAttentionState::new(config, &device)?;
        let stream_logits = model_forward_paged_streaming_with(
            &backend,
            token_ids,
            &weights,
            config,
            &mut stream_cache,
            &stream_bt,
            0,
            Some(&mut stream_state),
            None,
            tile_size,
            false,
            None,
        )?;

        Ok((mono_logits, stream_logits))
    }

    /// Compare the streaming last-tile full logits against the matching
    /// slice of the monolithic logits.
    fn assert_last_tile_matches(
        mono_logits: &Tensor,
        stream_logits: &Tensor,
        total_len: usize,
        tile_size: usize,
        tol: f32,
    ) -> Result<()> {
        // Last tile spans [last_start, total_len).
        let last_start = total_len - ((total_len - 1) % tile_size + 1);
        let last_len = total_len - last_start;
        let mono_slice = mono_logits
            .narrow(1, last_start, last_len)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let stream_slice = stream_logits.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            mono_slice.len(),
            stream_slice.len(),
            "last tile length mismatch"
        );
        let mut max_abs = 0f32;
        for (a, b) in mono_slice.iter().zip(stream_slice.iter()) {
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        assert!(
            max_abs <= tol,
            "streaming vs monolithic max_abs_diff={max_abs:e} exceeds {tol:e}"
        );
        Ok(())
    }

    #[test]
    fn test_streaming_matches_monolithic_cpu_small() -> Result<()> {
        let config = streaming_test_config();
        let total = 128;
        let tile = 64;
        let tokens = deterministic_tokens(total, config.vocab_size as u32);
        let (mono, stream) = run_parity(&config, &tokens, tile, 64)?;
        assert_eq!(mono.dims(), &[1, total, config.vocab_size]);
        assert_eq!(stream.dims(), &[1, tile, config.vocab_size]);
        assert_last_tile_matches(&mono, &stream, total, tile, 1e-5)?;
        Ok(())
    }

    #[test]
    fn test_streaming_last_hidden_matches_monolithic_cpu() -> Result<()> {
        let config = streaming_test_config();
        let device = Device::Cpu;
        let total = GDN_CHUNK_SIZE * 2 + 7;
        let tile = GDN_CHUNK_SIZE;
        let tokens = deterministic_tokens(total, config.vocab_size as u32);
        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = test_backend(&device);

        let (mut mono_cache, mono_bt) = make_paged_setup(&config, total, 64, &device)?;
        let mut mono_state = LinearAttentionState::new(&config, &device)?;
        let (mono_logits, mono_hidden) = model_forward_paged_last_token_with_last_hidden(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut mono_cache,
            &mono_bt,
            0,
            Some(&mut mono_state),
            None,
            None,
        )?;

        let (mut stream_cache, stream_bt) = make_paged_setup(&config, total, 64, &device)?;
        let mut stream_state = LinearAttentionState::new(&config, &device)?;
        let (stream_logits, stream_hidden) =
            model_forward_paged_streaming_last_token_with_last_hidden_with(
                &backend,
                &tokens,
                &weights,
                &config,
                &mut stream_cache,
                &stream_bt,
                0,
                Some(&mut stream_state),
                None,
                tile,
            )?;

        assert_eq!(stream_logits.dims(), &[1, 1, config.vocab_size]);
        assert_eq!(stream_hidden.dims(), &[1, 1, config.hidden_size]);
        let logits_diff = (&mono_logits - &stream_logits)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let hidden_diff = (&mono_hidden - &stream_hidden)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            logits_diff <= 1e-5,
            "streaming MTP prefill logits drifted: max_abs_diff={logits_diff:e}"
        );
        assert!(
            hidden_diff <= 1e-5,
            "streaming MTP prefill h_prev drifted: max_abs_diff={hidden_diff:e}"
        );
        Ok(())
    }

    #[test]
    fn test_streaming_matches_monolithic_cpu_mid() -> Result<()> {
        let config = streaming_test_config();
        let total = 512;
        let tile = 128;
        let tokens = deterministic_tokens(total, config.vocab_size as u32);
        let (mono, stream) = run_parity(&config, &tokens, tile, 64)?;
        assert_eq!(mono.dims(), &[1, total, config.vocab_size]);
        assert_eq!(stream.dims(), &[1, tile, config.vocab_size]);
        assert_last_tile_matches(&mono, &stream, total, tile, 1e-5)?;
        Ok(())
    }

    #[test]
    fn test_streaming_tile_invariance_cpu() -> Result<()> {
        // For a fixed token sequence, the last token's logits must agree
        // across every legal tile size (multiples of GDN_CHUNK_SIZE that
        // divide or partition `total`). The monolithic run is the reference;
        // every tile size collapses to the same final-token logits.
        let config = streaming_test_config();
        let total = 256;
        let tokens = deterministic_tokens(total, config.vocab_size as u32);

        // Monolithic reference: take the last row of [1, total, V] logits.
        let device = Device::Cpu;
        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = test_backend(&device);
        let (mut mono_cache, mono_bt) = make_paged_setup(&config, total, 64, &device)?;
        let mut mono_state = LinearAttentionState::new(&config, &device)?;
        let mono_logits = model_forward_paged(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut mono_cache,
            &mono_bt,
            0,
            Some(&mut mono_state),
            None,
            None,
        )?;
        let reference_last = mono_logits
            .narrow(1, total - 1, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        for tile in [64usize, 128, 256] {
            let (mut cache, bt) = make_paged_setup(&config, total, 64, &device)?;
            let mut state = LinearAttentionState::new(&config, &device)?;
            let logits = model_forward_paged_streaming_with(
                &backend,
                &tokens,
                &weights,
                &config,
                &mut cache,
                &bt,
                0,
                Some(&mut state),
                None,
                tile,
                true, // last_token_only — matches production dispatch
                None,
            )?;
            assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
            let last = logits.flatten_all()?.to_vec1::<f32>()?;
            let max_abs = reference_last
                .iter()
                .zip(last.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs <= 1e-5,
                "tile={tile} last-token max_abs_diff={max_abs:e} exceeds 1e-5"
            );
        }
        Ok(())
    }

    #[test]
    fn test_model_forward_paged_last_token_matches_full_last_row_cpu() -> Result<()> {
        let config = streaming_test_config();
        let total = 128;
        let tokens = deterministic_tokens(total, config.vocab_size as u32);
        let device = Device::Cpu;
        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = test_backend(&device);

        let (mut full_cache, full_bt) = make_paged_setup(&config, total, 64, &device)?;
        let mut full_state = LinearAttentionState::new(&config, &device)?;
        let full_logits = model_forward_paged(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut full_cache,
            &full_bt,
            0,
            Some(&mut full_state),
            None,
            None,
        )?;
        let reference_last = full_logits
            .narrow(1, total - 1, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let (mut last_cache, last_bt) = make_paged_setup(&config, total, 64, &device)?;
        let mut last_state = LinearAttentionState::new(&config, &device)?;
        let last_logits = model_forward_paged_last_token(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut last_cache,
            &last_bt,
            0,
            Some(&mut last_state),
            None,
            None,
        )?;
        assert_eq!(last_logits.dims(), &[1, 1, config.vocab_size]);
        let last = last_logits.flatten_all()?.to_vec1::<f32>()?;
        let max_abs = reference_last
            .iter()
            .zip(last.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs <= 1e-5,
            "last-token prefill max_abs_diff={max_abs:e} exceeds 1e-5"
        );

        let expected_token = crate::sampling::greedy_sample(&last_logits)?;
        let (mut greedy_cache, greedy_bt) = make_paged_setup(&config, total, 64, &device)?;
        let mut greedy_state = LinearAttentionState::new(&config, &device)?;
        let greedy_token = model_forward_paged_last_token_greedy(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut greedy_cache,
            &greedy_bt,
            0,
            Some(&mut greedy_state),
            None,
            None,
        )?;
        assert_eq!(
            greedy_token, expected_token,
            "last-token greedy prefill should match greedy_sample(last-token logits)"
        );

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_weighted_lm_head_prep_argmax_matches_final_rmsnorm_argmax_metal() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };

        unsafe {
            std::env::remove_var("KILN_DISABLE_WEIGHTED_LM_HEAD_PREP");
        }

        let hidden = 128usize;
        let vocab = 257usize;
        let best = 42usize;
        let x_data: Vec<f32> = (0..hidden)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.0234375)
            .collect();
        let norm_weight_data: Vec<f32> = (0..hidden)
            .map(|i| 0.75 + (i % 17) as f32 * 0.015625)
            .collect();
        let mut weight_data: Vec<f32> = (0..(hidden * vocab))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.0009765625)
            .collect();
        for i in 0..hidden {
            weight_data[i * vocab + best] = x_data[i] * norm_weight_data[i];
        }

        let x = Tensor::from_slice(&x_data, (1usize, 1usize, hidden), &device)?
            .to_dtype(DType::BF16)?;
        let norm_weight =
            Tensor::from_slice(&norm_weight_data, (hidden,), &device)?.to_dtype(DType::BF16)?;
        let weight_t =
            Tensor::from_slice(&weight_data, (hidden, vocab), &device)?.to_dtype(DType::BF16)?;

        let normed = rms_norm(&x, &norm_weight, 1e-6)?;
        let reference = lm_head_argmax(&normed, &weight_t)?;
        let weighted = lm_head_weighted_prep_argmax(&x, &norm_weight, &weight_t)?
            .context("weighted lm-head prep should support Metal BF16 [1,1,H]")?;

        assert_eq!(reference as usize, best);
        assert_eq!(weighted, reference);
        Ok(())
    }

    #[test]
    fn test_streaming_preserves_state_cpu() -> Result<()> {
        // After prefill, run a single decode step on top of the resulting
        // (paged_cache, linear_state). If state was preserved bit-exact
        // across tile boundaries, the decode-token logits must agree with
        // the monolithic reference.
        let config = streaming_test_config();
        let total = 192;
        let tile = 64;
        let tokens = deterministic_tokens(total, config.vocab_size as u32);
        let next_token: u32 = 11;

        let device = Device::Cpu;
        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = test_backend(&device);

        // Monolithic prefill, then 1 decode step.
        let (mut mono_cache, mono_bt) = make_paged_setup(&config, total + 1, 64, &device)?;
        let mut mono_state = LinearAttentionState::new(&config, &device)?;
        let _ = model_forward_paged(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut mono_cache,
            &mono_bt,
            0,
            Some(&mut mono_state),
            None,
            None,
        )?;
        let mono_decode = model_forward_paged(
            &backend,
            &[next_token],
            &weights,
            &config,
            &mut mono_cache,
            &mono_bt,
            total,
            Some(&mut mono_state),
            None,
            None,
        )?;

        // Streaming prefill, then 1 decode step.
        let (mut stream_cache, stream_bt) = make_paged_setup(&config, total + 1, 64, &device)?;
        let mut stream_state = LinearAttentionState::new(&config, &device)?;
        let _ = model_forward_paged_streaming_with(
            &backend,
            &tokens,
            &weights,
            &config,
            &mut stream_cache,
            &stream_bt,
            0,
            Some(&mut stream_state),
            None,
            tile,
            true,
            None,
        )?;
        let stream_decode = model_forward_paged(
            &backend,
            &[next_token],
            &weights,
            &config,
            &mut stream_cache,
            &stream_bt,
            total,
            Some(&mut stream_state),
            None,
            None,
        )?;

        let a = mono_decode.flatten_all()?.to_vec1::<f32>()?;
        let b = stream_decode.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(a.len(), b.len());
        let max_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs <= 1e-5,
            "decode-after-streaming max_abs_diff={max_abs:e} exceeds 1e-5 \
             (state was not bit-exact preserved across tile boundaries)"
        );
        Ok(())
    }

    /// Phase 10 — training-time streaming GDN parity (CPU).
    ///
    /// Direct unit test of [`gated_deltanet_forward_streaming`] against the
    /// monolithic [`gated_deltanet_forward`] on a small GDN-only input. Both
    /// paths must produce equal output tensors and equal final state.
    ///
    /// This test does NOT touch any `KILN_STREAMING_PREFILL` env vars so it
    /// is safe under multi-threaded `cargo test`; the env-driven dispatch
    /// inside `model_forward_segment` is exercised separately by
    /// `test_model_forward_segment_streaming_matches_monolithic_cpu` (which
    /// relies on nextest per-test process isolation).
    #[test]
    fn test_gated_deltanet_forward_streaming_matches_monolithic_cpu() -> Result<()> {
        let config = streaming_test_config();
        let device = Device::Cpu;
        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = test_backend(&device);

        // Pull the first GDN layer (layer 0 — full_attention_interval=4 so
        // layers 0,1,2 are GDN and layer 3 is full-attn).
        let lin_weights = match &weights.layers[0].attention {
            GpuAttentionWeights::Linear(w) => w,
            GpuAttentionWeights::Full(_) => panic!("test setup error: layer 0 must be GDN"),
        };

        // Deterministic input. T must be a multiple of GDN_CHUNK_SIZE so
        // both monolithic and tiled paths exercise the chunkwise kernel.
        let total = GDN_CHUNK_SIZE * 3; // 192 tokens
        let tile = GDN_CHUNK_SIZE; // 64-token tiles -> 3 tiles
        let n: usize = total * config.hidden_size;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.013).sin()) * 0.1).collect();
        let x = Tensor::new(data, &device)?.reshape((1, total, config.hidden_size))?;

        // Monolithic.
        let mut mono_state = LinearAttentionState::new(&config, &device)?;
        let mono_out = gated_deltanet_forward(
            &backend,
            &x,
            lin_weights,
            &config,
            &mut mono_state.recurrent_states[0],
            &mut mono_state.conv_states[0],
            false,
            false,
            None,
        )?;

        // Streaming/tiled.
        let mut stream_state = LinearAttentionState::new(&config, &device)?;
        let stream_out = gated_deltanet_forward_streaming(
            &backend,
            &x,
            lin_weights,
            &config,
            &mut stream_state.recurrent_states[0],
            &mut stream_state.conv_states[0],
            tile,
            None,
        )?;

        assert_eq!(mono_out.dims(), stream_out.dims());
        let mono_v = mono_out.flatten_all()?.to_vec1::<f32>()?;
        let stream_v = stream_out.flatten_all()?.to_vec1::<f32>()?;
        let max_abs_out = mono_v
            .iter()
            .zip(stream_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs_out <= 1e-5,
            "streaming GDN output drifted from monolithic: max_abs_diff={max_abs_out:e}"
        );

        // Final recurrent state must match (the load-bearing invariant for
        // training-time streaming — autograd flows through this state thread).
        let mr = mono_state.recurrent_states[0]
            .flatten_all()?
            .to_vec1::<f32>()?;
        let sr = stream_state.recurrent_states[0]
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(mr.len(), sr.len());
        let max_abs_recur = mr
            .iter()
            .zip(sr.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs_recur <= 1e-5,
            "streaming GDN recurrent state drifted: max_abs_diff={max_abs_recur:e}"
        );

        // Final conv state must match (drives correctness of any subsequent
        // decode step that consumes it).
        let mc = mono_state.conv_states[0].flatten_all()?.to_vec1::<f32>()?;
        let sc = stream_state.conv_states[0]
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(mc.len(), sc.len());
        let max_abs_conv = mc
            .iter()
            .zip(sc.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs_conv <= 1e-5,
            "streaming GDN conv state drifted: max_abs_diff={max_abs_conv:e}"
        );

        Ok(())
    }

    /// Phase 10 — training-time streaming GDN parity for `model_forward_segment`.
    ///
    /// Runs `model_forward_segment` over the full layer stack twice on the
    /// same input: once monolithic (env unset), once with
    /// `KILN_STREAMING_PREFILL=1` and `KILN_STREAMING_TILE_TOKENS=64` so the
    /// 192-token input is split into 3 tiles. The two outputs must match
    /// within FP32 tolerance and the final per-layer state must match.
    ///
    /// Relies on nextest per-test process isolation for safe env-var
    /// manipulation; `cargo nextest run` is the canonical kiln test runner
    /// (see `crates/kiln-model/src/forward.rs` `test_streaming_prefill_env_helpers`).
    #[test]
    fn test_model_forward_segment_streaming_matches_monolithic_cpu() -> Result<()> {
        let config = streaming_test_config();
        let device = Device::Cpu;
        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = test_backend(&device);

        let total = GDN_CHUNK_SIZE * 3; // 192 tokens
        let tile = GDN_CHUNK_SIZE; // 64-token tiles -> 3 tiles
        let n: usize = total * config.hidden_size;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.017).cos()) * 0.1).collect();
        let hidden = Tensor::new(data, &device)?.reshape((1, total, config.hidden_size))?;
        let positions: Vec<u32> = (0..total as u32).collect();

        // Monolithic — env vars unset for this thread/process.
        unsafe {
            std::env::remove_var("KILN_STREAMING_PREFILL");
            std::env::remove_var("KILN_STREAMING_TILE_TOKENS");
        }
        let mut mono_state = LinearAttentionState::new(&config, &device)?;
        let mono_out = model_forward_segment(
            &backend,
            hidden.clone(),
            &weights,
            &config,
            &positions,
            0,
            config.num_layers,
            Some(&mut mono_state),
            None,
        )?;

        // Streaming — env vars set so streaming_prefill_enabled_for(Cpu, T)
        // returns true and tile_size = 64.
        unsafe {
            std::env::set_var("KILN_STREAMING_PREFILL", "1");
            std::env::set_var("KILN_STREAMING_TILE_TOKENS", tile.to_string());
        }
        let mut stream_state = LinearAttentionState::new(&config, &device)?;
        let stream_out = model_forward_segment(
            &backend,
            hidden.clone(),
            &weights,
            &config,
            &positions,
            0,
            config.num_layers,
            Some(&mut stream_state),
            None,
        )?;
        // Restore for subsequent tests in this process (best-effort).
        unsafe {
            std::env::remove_var("KILN_STREAMING_PREFILL");
            std::env::remove_var("KILN_STREAMING_TILE_TOKENS");
        }

        assert_eq!(mono_out.dims(), stream_out.dims());
        let mv = mono_out.flatten_all()?.to_vec1::<f32>()?;
        let sv = stream_out.flatten_all()?.to_vec1::<f32>()?;
        let max_abs = mv
            .iter()
            .zip(sv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "model_forward_segment streaming output drifted from monolithic: max_abs_diff={max_abs:e}"
        );

        for (l, (m, s)) in mono_state
            .recurrent_states
            .iter()
            .zip(stream_state.recurrent_states.iter())
            .enumerate()
        {
            let mv = m.flatten_all()?.to_vec1::<f32>()?;
            let sv = s.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(mv.len(), sv.len(), "recurrent_states[{l}] length mismatch");
            let max_abs = mv
                .iter()
                .zip(sv.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs <= 1e-4,
                "model_forward_segment streaming recurrent_states[{l}] drifted: max_abs_diff={max_abs:e}"
            );
        }
        for (l, (m, s)) in mono_state
            .conv_states
            .iter()
            .zip(stream_state.conv_states.iter())
            .enumerate()
        {
            let mv = m.flatten_all()?.to_vec1::<f32>()?;
            let sv = s.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(mv.len(), sv.len(), "conv_states[{l}] length mismatch");
            let max_abs = mv
                .iter()
                .zip(sv.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs <= 1e-4,
                "model_forward_segment streaming conv_states[{l}] drifted: max_abs_diff={max_abs:e}"
            );
        }

        Ok(())
    }

    /// CUDA parity for streaming/tiled GDN prefill.
    ///
    /// Mirrors `test_streaming_matches_monolithic_cpu_mid` but on CUDA at
    /// T=2048, tile=512 (the configuration the Phase 7 GPU spike validates).
    /// Asserts (1) full-tile logits match the matching slice of the
    /// monolithic logits, and (2) `LinearAttentionState.recurrent_states[l]`
    /// and `state.conv_states[l]` are equal across the two paths after
    /// prefill — the state hand-off is the load-bearing part of streaming.
    ///
    /// Tolerance: 1e-4. The design doc (PROFILING.md §c "CUDA parity")
    /// argues bit-exactness is achievable because GDN recurrent state stays
    /// in F32 and the conv1d F32 promotion makes the conv path
    /// deterministic. In practice, candle CUDA matmul reduction order can
    /// vary with shape, so we use a small FP32 tolerance rather than
    /// strict equality.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_streaming_matches_monolithic_cuda() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available, skipping test_streaming_matches_monolithic_cuda");
                return Ok(());
            }
        };

        let config = streaming_test_config();
        let total = 2048usize;
        let tile = 512usize;
        let block_size = 64usize; // == GDN_CHUNK_SIZE
        let tokens = deterministic_tokens(total, config.vocab_size as u32);

        let weights = make_hybrid_gpu_weights(
            &device,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_layers,
            config.full_attention_interval,
        )?;
        let backend = crate::backend::for_device(&device);

        // Monolithic: single forward pass, full LM head.
        let (mut mono_cache, mono_bt) = make_paged_setup(&config, total, block_size, &device)?;
        let mut mono_state = LinearAttentionState::new(&config, &device)?;
        let mono_logits = model_forward_paged(
            &*backend,
            &tokens,
            &weights,
            &config,
            &mut mono_cache,
            &mono_bt,
            0,
            Some(&mut mono_state),
            None,
            None,
        )?;

        // Streaming: tiled prefill, last_token_only=false so we get a full
        // last-tile logits slice for row-by-row comparison.
        let (mut stream_cache, stream_bt) = make_paged_setup(&config, total, block_size, &device)?;
        let mut stream_state = LinearAttentionState::new(&config, &device)?;
        let stream_logits = model_forward_paged_streaming_with(
            &*backend,
            &tokens,
            &weights,
            &config,
            &mut stream_cache,
            &stream_bt,
            0,
            Some(&mut stream_state),
            None,
            tile,
            false,
            None,
        )?;

        assert_eq!(mono_logits.dims(), &[1, total, config.vocab_size]);
        assert_eq!(stream_logits.dims(), &[1, tile, config.vocab_size]);

        // (1) Last-tile logits parity.
        assert_last_tile_matches(&mono_logits, &stream_logits, total, tile, 1e-4)?;

        // (2) Per-layer state parity (recurrent + conv).
        assert_eq!(
            mono_state.recurrent_states.len(),
            stream_state.recurrent_states.len(),
            "recurrent_states layer count mismatch"
        );
        assert_eq!(
            mono_state.conv_states.len(),
            stream_state.conv_states.len(),
            "conv_states layer count mismatch"
        );
        for (l, (m, s)) in mono_state
            .recurrent_states
            .iter()
            .zip(stream_state.recurrent_states.iter())
            .enumerate()
        {
            let m_v = m.flatten_all()?.to_vec1::<f32>()?;
            let s_v = s.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(
                m_v.len(),
                s_v.len(),
                "recurrent_states[{l}] length mismatch"
            );
            let max_abs = m_v
                .iter()
                .zip(s_v.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs <= 1e-4,
                "recurrent_states[{l}] max_abs_diff={max_abs:e} exceeds 1e-4"
            );
        }
        for (l, (m, s)) in mono_state
            .conv_states
            .iter()
            .zip(stream_state.conv_states.iter())
            .enumerate()
        {
            let m_v = m.flatten_all()?.to_vec1::<f32>()?;
            let s_v = s.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(m_v.len(), s_v.len(), "conv_states[{l}] length mismatch");
            let max_abs = m_v
                .iter()
                .zip(s_v.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs <= 1e-4,
                "conv_states[{l}] max_abs_diff={max_abs:e} exceeds 1e-4"
            );
        }

        Ok(())
    }

    #[test]
    fn test_streaming_prefill_env_helpers() {
        // Each nextest test runs in its own process, so env-var manipulation
        // here is safe. We verify the dispatch helpers return what
        // `model_forward_paged_streaming` reads from the environment.
        unsafe {
            std::env::remove_var("KILN_STREAMING_PREFILL");
            std::env::remove_var("KILN_STREAMING_PREFILL_THRESHOLD_TOKENS");
            std::env::remove_var("KILN_STREAMING_TILE_TOKENS");
            std::env::remove_var("KILN_STREAMING_LAST_TOKEN_LM_HEAD");
        }
        assert!(!streaming_prefill_enabled(), "default must be disabled");
        assert!(!streaming_prefill_default_for(
            StreamingPrefillDeviceKind::Cpu,
            STREAMING_PREFILL_CUDA_DEFAULT_THRESHOLD
        ));
        assert!(!streaming_prefill_default_for(
            StreamingPrefillDeviceKind::Cuda,
            STREAMING_PREFILL_CUDA_DEFAULT_THRESHOLD - 1
        ));
        assert!(streaming_prefill_default_for(
            StreamingPrefillDeviceKind::Cuda,
            STREAMING_PREFILL_CUDA_DEFAULT_THRESHOLD
        ));
        assert!(streaming_prefill_default_for(
            StreamingPrefillDeviceKind::Cuda,
            12_000
        ));
        assert!(streaming_prefill_default_for(
            StreamingPrefillDeviceKind::Cuda,
            43_814
        ));
        assert!(!streaming_prefill_default_for(
            StreamingPrefillDeviceKind::Metal,
            STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD - 1
        ));
        assert!(streaming_prefill_default_for(
            StreamingPrefillDeviceKind::Metal,
            STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
        ));
        assert_eq!(
            streaming_prefill_threshold_tokens(),
            STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
        );
        assert!(!streaming_prefill_enabled_for(
            &Device::Cpu,
            STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
        ));
        assert_eq!(streaming_tile_tokens(), STREAMING_PREFILL_DEFAULT_TILE);
        assert_eq!(
            streaming_tile_tokens_for(&Device::Cpu),
            STREAMING_PREFILL_DEFAULT_TILE
        );
        assert!(streaming_last_token_lm_head(), "default must be true");

        #[cfg(feature = "metal")]
        if let Some(device) = crate::backend::metal::try_new_metal() {
            assert!(!streaming_prefill_enabled_for(
                &device,
                STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD - 1
            ));
            assert!(streaming_prefill_enabled_for(
                &device,
                STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
            ));
            assert_eq!(
                streaming_tile_tokens_for(&device),
                STREAMING_PREFILL_METAL_DEFAULT_TILE
            );
            unsafe {
                std::env::set_var("KILN_STREAMING_PREFILL_THRESHOLD_TOKENS", "1024");
            }
            assert_eq!(streaming_prefill_threshold_tokens(), 1024);
            assert!(!streaming_prefill_default_for(
                StreamingPrefillDeviceKind::Metal,
                1023
            ));
            assert!(streaming_prefill_default_for(
                StreamingPrefillDeviceKind::Metal,
                1024
            ));
            assert!(!streaming_prefill_enabled_for(&device, 1023));
            assert!(streaming_prefill_enabled_for(&device, 1024));
        }

        unsafe {
            std::env::set_var("KILN_STREAMING_PREFILL", "1");
        }
        assert!(streaming_prefill_enabled());
        assert!(streaming_prefill_enabled_for(&Device::Cpu, 1));

        unsafe {
            std::env::set_var("KILN_STREAMING_PREFILL", "0");
        }
        assert!(!streaming_prefill_enabled());
        assert!(!streaming_prefill_enabled_for(
            &Device::Cpu,
            STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
        ));

        unsafe {
            std::env::set_var("KILN_STREAMING_PREFILL_THRESHOLD_TOKENS", "0");
        }
        assert_eq!(
            streaming_prefill_threshold_tokens(),
            STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
        );

        unsafe {
            std::env::set_var("KILN_STREAMING_TILE_TOKENS", "256");
        }
        assert_eq!(streaming_tile_tokens(), 256);
        assert_eq!(streaming_tile_tokens_for(&Device::Cpu), 256);

        // Bad value (not a multiple of GDN_CHUNK_SIZE) falls back to default.
        unsafe {
            std::env::set_var("KILN_STREAMING_TILE_TOKENS", "65");
        }
        assert_eq!(streaming_tile_tokens(), STREAMING_PREFILL_DEFAULT_TILE);

        unsafe {
            std::env::set_var("KILN_STREAMING_LAST_TOKEN_LM_HEAD", "0");
        }
        assert!(!streaming_last_token_lm_head());

        // Cleanup so this test does not leak state to peers (defensive even
        // though nextest isolates by process).
        unsafe {
            std::env::remove_var("KILN_STREAMING_PREFILL");
            std::env::remove_var("KILN_STREAMING_TILE_TOKENS");
            std::env::remove_var("KILN_STREAMING_LAST_TOKEN_LM_HEAD");
        }
    }
}
