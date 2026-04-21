//! Optional per-step instrumentation for native MTP (Multi-Token Prediction)
//! speculative decode.
//!
//! Off by default — every call site early-outs cheaply when `KILN_MTP_DEBUG`
//! is unset. Enable with `KILN_MTP_DEBUG=1` to emit one `tracing::info!` line
//! per draft and per verify pass containing:
//!
//! - `h_prev` L2 norm at MTP draft entry (sanity check on tensor magnitudes)
//! - top-K MTP draft logits as `(token_id, logit)` pairs (default K=5)
//! - top-K base verify-pos0 logits (the prediction the draft is graded against)
//! - the chosen draft token, the target token at pos 0, and accept/reject
//!
//! By default only the first `KILN_MTP_DEBUG_MAX_CALLS` (default 16) draft
//! steps log; set the env var to `0` for unlimited. The counter is a single
//! process-wide `AtomicUsize`, so a fresh process resets to zero. Use
//! [`reset_counter`] to re-arm without restarting (handy from tests).
//!
//! This module exists to support Phase B of the low-α root-cause investigation
//! tracked in `PROFILING.md` ("MTP α=0.411 vs 0.72 floor"). The intended
//! workflow is: enable on a pod, capture a 16-step trace, diff against an HF
//! reference run on the same prompt, and either ship a Tier 1 fix (if a
//! tensor or layout bug appears) or document the runtime evidence in a Tier 2
//! diagnostic update to `PROFILING.md`.

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use candle_core::{DType, Tensor};

static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

const DEFAULT_MAX_CALLS: usize = 16;

/// True when `KILN_MTP_DEBUG=1` (or `true`) is set.
pub fn is_enabled() -> bool {
    std::env::var("KILN_MTP_DEBUG")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Returns true if instrumentation is enabled AND we are still under the
/// per-process call cap. Increments the call counter as a side effect, so each
/// invocation moves the budget forward by one.
///
/// The cap defaults to 16 to keep traces small enough to read by eye; set
/// `KILN_MTP_DEBUG_MAX_CALLS=0` for unlimited (noisy but useful when chasing
/// a specific reproducer that needs many tokens).
pub fn should_log() -> bool {
    if !is_enabled() {
        return false;
    }
    let max = std::env::var("KILN_MTP_DEBUG_MAX_CALLS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_CALLS);
    if max == 0 {
        return true;
    }
    let n = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
    n < max
}

/// Reset the per-process call counter to zero. Test-only convenience; not
/// reachable from outside this crate's instrumentation paths.
pub fn reset_counter() {
    CALL_COUNT.store(0, Ordering::Relaxed);
}

/// Extract the top-K `(token_id, logit)` pairs from a flattenable logits
/// tensor. Accepts shapes `[V]`, `[1, V]`, or `[1, 1, V]` — anything that
/// flattens to a 1-D vocab-sized vector.
pub fn top_k_logits(logits: &Tensor, k: usize) -> Result<Vec<(u32, f32)>> {
    let vals = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let mut idx: Vec<(u32, f32)> = vals
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    Ok(idx)
}

/// Compute the L2 norm of a tensor as a single f32. Diagnostic check on
/// tensor magnitudes — useful for catching the "all zeros" or "explosion"
/// failure modes that would silently kill draft accuracy.
pub fn tensor_l2_norm(t: &Tensor) -> Result<f32> {
    let v = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok(v.iter().map(|x| x * x).sum::<f32>().sqrt())
}

/// Format a top-K list as `"[(id=42, l=12.34), (id=1, l=11.20), ...]"`.
pub fn format_top_k(top: &[(u32, f32)]) -> String {
    let parts: Vec<String> = top
        .iter()
        .map(|(id, l)| format!("(id={}, l={:.3})", id, l))
        .collect();
    format!("[{}]", parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn top_k_picks_largest_in_descending_order() {
        let t = Tensor::new(&[1.0_f32, 5.0, 3.0, 4.0, 2.0][..], &Device::Cpu).unwrap();
        let top = top_k_logits(&t, 3).unwrap();
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 1); // logit 5.0
        assert_eq!(top[1].0, 3); // logit 4.0
        assert_eq!(top[2].0, 2); // logit 3.0
    }

    #[test]
    fn l2_norm_matches_hand_calculation() {
        let t = Tensor::new(&[3.0_f32, 4.0][..], &Device::Cpu).unwrap();
        let n = tensor_l2_norm(&t).unwrap();
        assert!((n - 5.0).abs() < 1e-5);
    }

    #[test]
    fn format_top_k_renders_compact_pairs() {
        let s = format_top_k(&[(7, 1.5), (2, 0.5)]);
        assert_eq!(s, "[(id=7, l=1.500), (id=2, l=0.500)]");
    }

    #[test]
    fn should_log_is_false_when_env_unset() {
        // SAFETY: tests are #[cfg(test)] and run with cargo's default serial
        // dispatch within a process. We only mutate KILN_MTP_DEBUG here, so a
        // single set/unset is fine for this check.
        unsafe {
            std::env::remove_var("KILN_MTP_DEBUG");
        }
        reset_counter();
        assert!(!should_log());
    }
}
