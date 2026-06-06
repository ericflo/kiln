//! Vulkan linear-dispatch policy helpers.
//!
//! `backend/vulkan.rs` owns the `BackendRuntime` facade. This module keeps the
//! per-submit safety policy for Vulkan-routed linear kernels next to the linear
//! concern without hiding the explicit VkTensor/buffer dispatch boundary.

use std::sync::OnceLock;

/// (#1082) Per-dispatch FLOP ceiling for the Vulkan-routed matmul.
///
/// Migrated inline from the deleted `backend::vulkan_linear_op` module
/// (its `candle_core::CustomOp1` training wrapper was removed when the kt
/// autograd tape became the sole grad producer). The forward-only FLCE
/// offset path in `linear_prefill_apply_offset` still needs the ceiling to
/// sub-chunk oversized dispatches: the host hard-hung twice on Strix Halo
/// when a single oversized submit (~4.36M workgroups) was queued, so the
/// ceiling caps per-submit FLOP. Tunable via `KILN_VULKAN_LINEAR_MAX_GFLOP`
/// (parsed once; `0` disables the guard).
const DEFAULT_MAX_FLOP_PER_DISPATCH: u64 = 20_000_000_000;

/// FLOP estimate for `[batch, hidden] @ [hidden, out_dim]` (one mul + one
/// add per inner term).
fn matmul_flop(batch: usize, hidden: usize, out_dim: usize) -> u64 {
    (batch as u64)
        .saturating_mul(hidden as u64)
        .saturating_mul(out_dim as u64)
        .saturating_mul(2)
}

fn max_flop_per_dispatch() -> u64 {
    static CEILING: OnceLock<u64> = OnceLock::new();
    *CEILING.get_or_init(|| {
        std::env::var("KILN_VULKAN_LINEAR_MAX_GFLOP")
            .ok()
            .as_deref()
            .map(str::trim)
            .and_then(|s| s.parse::<f64>().ok())
            .map(|gflop| {
                if gflop <= 0.0 {
                    u64::MAX
                } else {
                    (gflop * 1.0e9_f64).round() as u64
                }
            })
            .unwrap_or(DEFAULT_MAX_FLOP_PER_DISPATCH)
    })
}

/// True when the requested matmul shape would exceed the per-dispatch FLOP
/// ceiling; the caller sub-chunks via [`max_chunk_dim_for_flop`].
pub(super) fn dispatch_exceeds_safety_ceiling(batch: usize, hidden: usize, out_dim: usize) -> bool {
    matmul_flop(batch, hidden, out_dim) > max_flop_per_dispatch()
}

/// Largest `chunk_dim` such that `2 x other_dim_product x chunk_dim <=
/// max_flop_per_dispatch()`. Always >= 1; returns `usize::MAX` when the
/// guard is disabled.
pub(super) fn max_chunk_dim_for_flop(other_dim_product: usize) -> usize {
    let max_flop = max_flop_per_dispatch();
    if max_flop == u64::MAX {
        return usize::MAX;
    }
    let denom = (other_dim_product as u64).saturating_mul(2).max(1);
    let chunk = (max_flop / denom) as usize;
    chunk.max(1)
}
