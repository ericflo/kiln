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

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};

static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
static DUMP_DONE: AtomicBool = AtomicBool::new(false);

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

/// True when `KILN_MTP_SWAP_FC_NORMS=1` (or `true`) is set. When enabled,
/// `mtp_forward_step` swaps which RMSNorm weight is applied to which half of
/// the `fc` input — i.e. `pre_fc_norm_hidden` is applied to the embedding
/// half and `pre_fc_norm_embedding` is applied to the `h_prev` half. This is
/// the Phase B2 secondary-hypothesis A/B test from `PROFILING.md`: if α
/// jumps with the swap on, the loader pairs the wrong norm tensor to the
/// wrong half of the fused `fc` input. Read on every call (no caching) so
/// runs can A/B by toggling the env var between processes.
pub fn is_swap_fc_norms_enabled() -> bool {
    std::env::var("KILN_MTP_SWAP_FC_NORMS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Path to which `mtp_forward_step` should dump all 8 Phase B6 numerical
/// bisect taps on the first draft call, if set. Unset → no dump. Set →
/// after the first call with valid tensors, the file is written and any
/// subsequent call is a no-op (see `try_consume_dump_slot`).
///
/// Paired with a Python reference implementation (see
/// `scripts/mtp_reference_dump.py`) that produces an identically-named
/// safetensors file for the same fixed prompt + seed. A post-hoc
/// per-tap `allclose` sweep (see `scripts/mtp_compare.py`) localizes
/// the first divergence and narrows the remaining runtime hypotheses.
pub fn dump_path() -> Option<String> {
    std::env::var("KILN_MTP_DUMP_PATH").ok().filter(|s| !s.is_empty())
}

/// First call after the path env var is set returns `true`; all later
/// calls return `false` so we only write the file once. Idempotent across
/// the whole process.
pub fn try_consume_dump_slot() -> bool {
    if dump_path().is_none() {
        return false;
    }
    DUMP_DONE
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
}

/// Reset the dump latch. Test-only helper.
#[cfg(test)]
fn reset_dump_latch() {
    DUMP_DONE.store(false, Ordering::SeqCst);
}

/// Convert a tensor to a contiguous host float32 `Vec<f32>` plus shape.
/// Used by [`write_mtp_dump`] to serialize taps uniformly regardless of
/// whether the source is BF16 on CUDA or F32 on CPU.
fn tensor_to_f32_host(t: &Tensor) -> Result<(Vec<usize>, Vec<f32>)> {
    let shape = t.dims().to_vec();
    let flat = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok((shape, flat))
}

/// Write the Phase B6 per-tap dump to the path from `KILN_MTP_DUMP_PATH`.
///
/// Output format is `safetensors` (same format kiln's loader already
/// supports, so the Python side can read it with the stock `safetensors`
/// package). All tensors are written as `F32` regardless of source dtype —
/// precision loss from BF16→F32 is one-directional (upcast) and keeps the
/// comparison numerically faithful.
///
/// The named taps correspond one-to-one with the 8 steps in the task brief:
///
/// | Tap name          | Meaning                                              |
/// | ----------------- | ---------------------------------------------------- |
/// | `h_main`          | Last-hidden-state from main model at position t.     |
/// | `tok_embed`       | `embed_tokens(draft_token_t-1)` (pre-norm).          |
/// | `fc_input`        | `concat([norm_emb, norm_h])` (input to `mtp.fc`).    |
/// | `fc_output`       | Post-`mtp.fc` linear projection.                     |
/// | `pre_layer`       | Input to the MTP inner transformer block (== `fc_output`). |
/// | `post_layer`      | Output of the MTP inner transformer block.           |
/// | `post_final_ln`   | Output of `mtp.final_layernorm` (pre-LM head).       |
/// | `mtp_logits`      | Output of `lm_head` on `post_final_ln`.              |
///
/// Plus metadata: `draft_token_id`, `mtp_pos`, `swap_fc_norms`.
pub fn write_mtp_dump(
    path: &str,
    draft_token_id: u32,
    mtp_pos: usize,
    swap_fc_norms: bool,
    taps: &[(&str, &Tensor)],
) -> Result<()> {
    use safetensors::tensor::{Dtype, TensorView};

    // Materialize every tensor to a host byte buffer first so the
    // TensorViews we build below can borrow from a stable backing store.
    let mut backings: Vec<(String, Vec<usize>, Dtype, Vec<u8>)> =
        Vec::with_capacity(taps.len() + 3);
    for (name, t) in taps {
        let (shape, flat) = tensor_to_f32_host(t)
            .with_context(|| format!("dump tap `{name}`: tensor→f32 host copy"))?;
        let mut bytes = Vec::with_capacity(flat.len() * 4);
        for v in &flat {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        backings.push(((*name).to_string(), shape, Dtype::F32, bytes));
    }

    // Metadata as 1-element I32 tensors so the Python side can read them
    // with one loader. Widened to i32 because safetensors 0.5 lacks U32 in
    // its serialization path on some builds; I32 is always available.
    let dti = draft_token_id as i32;
    backings.push((
        "meta__draft_token_id".into(),
        vec![1],
        Dtype::I32,
        dti.to_le_bytes().to_vec(),
    ));
    let mpi = mtp_pos as i32;
    backings.push((
        "meta__mtp_pos".into(),
        vec![1],
        Dtype::I32,
        mpi.to_le_bytes().to_vec(),
    ));
    let swf = if swap_fc_norms { 1i32 } else { 0i32 };
    backings.push((
        "meta__swap_fc_norms".into(),
        vec![1],
        Dtype::I32,
        swf.to_le_bytes().to_vec(),
    ));

    let mut views: Vec<(String, TensorView)> = Vec::with_capacity(backings.len());
    for (name, shape, dtype, bytes) in &backings {
        let view = TensorView::new(*dtype, shape.clone(), bytes.as_slice())
            .map_err(|e| anyhow::anyhow!("safetensors TensorView::new for `{name}`: {e:?}"))?;
        views.push((name.clone(), view));
    }

    let serialized = safetensors::serialize(views, &None)
        .map_err(|e| anyhow::anyhow!("safetensors::serialize MTP dump: {e:?}"))?;
    std::fs::write(path, serialized).with_context(|| format!("write MTP dump to {path}"))?;
    Ok(())
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

    #[test]
    fn dump_slot_fires_once_then_never() {
        // SAFETY: single-threaded test; scoped env mutation.
        let tmp = std::env::temp_dir().join("kiln_mtp_dump_slot_once.safetensors");
        let tmp_s = tmp.to_string_lossy().into_owned();
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_PATH", &tmp_s);
        }
        reset_dump_latch();
        assert_eq!(dump_path().as_deref(), Some(tmp_s.as_str()));
        assert!(try_consume_dump_slot());
        assert!(!try_consume_dump_slot());
        assert!(!try_consume_dump_slot());
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_PATH");
        }
    }

    #[test]
    fn write_and_reparse_mtp_dump_round_trips() {
        use safetensors::SafeTensors;

        let a = Tensor::new(&[1.0_f32, 2.0, 3.0, 4.0][..], &Device::Cpu).unwrap();
        let b = Tensor::new(&[0.5_f32, -0.25][..], &Device::Cpu).unwrap();
        let tmp = std::env::temp_dir().join("kiln_mtp_dump_round_trip.safetensors");
        let tmp_s = tmp.to_string_lossy().into_owned();
        write_mtp_dump(
            &tmp_s,
            /* draft_token_id = */ 561,
            /* mtp_pos = */ 0,
            /* swap_fc_norms = */ false,
            &[("h_main", &a), ("mtp_logits", &b)],
        )
        .unwrap();

        let raw = std::fs::read(&tmp).unwrap();
        let st = SafeTensors::deserialize(&raw).unwrap();
        let names: Vec<&str> = st.names().into_iter().map(|s| s.as_str()).collect();
        assert!(names.contains(&"h_main"));
        assert!(names.contains(&"mtp_logits"));
        assert!(names.contains(&"meta__draft_token_id"));
        assert!(names.contains(&"meta__mtp_pos"));
        assert!(names.contains(&"meta__swap_fc_norms"));

        let h = st.tensor("h_main").unwrap();
        assert_eq!(h.dtype(), safetensors::Dtype::F32);
        assert_eq!(h.shape(), &[4]);
        let meta = st.tensor("meta__draft_token_id").unwrap();
        assert_eq!(meta.dtype(), safetensors::Dtype::I32);
        let v = i32::from_le_bytes(meta.data()[0..4].try_into().unwrap());
        assert_eq!(v, 561);

        let _ = std::fs::remove_file(&tmp);
    }
}
